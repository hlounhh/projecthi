"""Export only the project's hash-verified models; INT8 needs real calibration images."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_files(directory, limit):
    """Stable, local image list; never substitute random/blank calibration tensors."""
    if limit < 1:
        raise ValueError("Calibration sample limit must be positive.")
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("Calibration directory does not exist.")
    paths = sorted(p for p in directory.rglob("*")
                   if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})
    if not paths:
        raise ValueError("INT8 requires representative real images; no images found.")
    # Spread a bounded subset across the sorted collection instead of taking its first clip.
    if len(paths) > limit:
        paths = [paths[round(i * (len(paths) - 1) / (limit - 1))]
                 for i in range(limit)] if limit > 1 else paths[:1]
    return paths


def preprocess(image, image_size):
    """Match YOLO static, square letterboxing and BGR->RGB normalization."""
    import numpy as np
    from ultralytics.data.augment import LetterBox

    image = LetterBox(new_shape=(image_size, image_size), auto=False)(image=image)
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def load_verified_model(task, image_size, threads, *, weights=None, source_sha256=None):
    if task == "pose":
        if weights is not None or source_sha256 is not None:
            raise ValueError("Only detect exports accept explicit pinned weights.")
        from ai.pose_detector import PoseDetector, POSE_WEIGHTS, POSE_SHA256
        detector = PoseDetector(image_size=image_size, threads=threads)
        return detector.model, POSE_WEIGHTS, POSE_SHA256
    if (weights is None) != (source_sha256 is None):
        raise ValueError("Explicit export needs both --weights and --source-sha256.")
    if weights is not None:
        from ai.posture_ensemble import load_pinned_checkpoint

        weights = Path(weights).resolve()
        if not weights.is_file():
            raise ValueError("Explicit export weights do not exist.")
        wrapper, _names = load_pinned_checkpoint(weights, source_sha256,
                                                 label=weights.parent.name,
                                                 image_size=image_size)
        return wrapper, weights, sha256(weights)
    from ai.posture_detector import PostureDetector, DEFAULT_WEIGHTS, EXPECTED_SHA256
    detector = PostureDetector(image_size=image_size, threads=threads)
    return detector.model, DEFAULT_WEIGHTS, EXPECTED_SHA256


def export_bundle(task, output, *, image_size=416, threads=2, int8=False,
                  calibration_images=None, calibration_limit=256,
                  weights=None, source_sha256=None, precisions=("fp32", "fp16")):
    if image_size not in (320, 416, 512, 640) or not 1 <= threads <= 8:
        raise ValueError("Use image size 320/416/512/640 and 1..8 CPU threads.")
    if not precisions or not set(precisions) <= {"fp32", "fp16"}:
        raise ValueError("Export precisions must be a non-empty subset of fp32/fp16.")
    if int8 and calibration_images is None:
        raise ValueError("INT8 requires --calibration-images with representative real images.")
    if calibration_images is not None and not int8:
        raise ValueError("--calibration-images requires --int8.")
    paths = calibration_files(calibration_images, calibration_limit) if int8 else []
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory to preserve existing exports.")
    output.parent.mkdir(parents=True, exist_ok=True)

    import openvino as ov
    import yaml

    model, source, source_digest = load_verified_model(task, image_size, threads,
                                                       weights=weights,
                                                       source_sha256=source_sha256)
    stage = Path(tempfile.mkdtemp(prefix=".export-", dir=output.parent))
    try:
        # Ultralytics chooses its export destination from pt_path. Point only the
        # in-memory model at our staging directory; the checkpoint stays untouched.
        model.model.pt_path = str(stage / "model.pt")
        # The older posture checkpoint predates this exporter-required attribute.
        model.model.task = task
        model.overrides["model"] = str(stage / "model.pt")
        onnx_path = Path(model.export(format="onnx", imgsz=image_size, batch=1,
                                      device="cpu", half=False, dynamic=False,
                                      simplify=False, opset=17, nms=False))
        if onnx_path.resolve().parent != stage:
            raise ValueError("Exporter returned an unexpected output directory.")
        metadata = {
            "description": "Local export of hash-verified project checkpoint",
            "task": task, "batch": 1, "imgsz": [image_size, image_size],
            "stride": int(model.model.stride.max()), "names": model.names,
            "dynamic": False, "args": {"nms": False},
        }
        if task == "pose":
            metadata["kpt_shape"] = list(model.model.model[-1].kpt_shape)
        converted = ov.convert_model(str(onnx_path))
        models = {"onnx": "model.onnx"}
        for precision, compress in (("fp32", False), ("fp16", True)):
            if precision not in precisions:
                continue
            folder = stage / f"{precision}_openvino_model"
            folder.mkdir()
            ov.save_model(converted, folder / "model.xml", compress_to_fp16=compress)
            (folder / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
            models[f"openvino_{precision}"] = folder.name
        calibration = []
        if int8:
            import cv2
            import nncf

            def transform(path):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Cannot decode calibration image: {path.name}")
                return preprocess(image, image_size)

            # Preflight every selected file so corrupt samples never silently reduce calibration.
            for path in paths:
                transform(path)
                calibration.append({"name": path.name, "sha256": sha256(path)})
            dataset = nncf.Dataset(paths, transform)
            quantized = nncf.quantize(converted, dataset, subset_size=len(paths),
                                      preset=nncf.QuantizationPreset.MIXED,
                                      target_device=nncf.TargetDevice.CPU)
            folder = stage / "int8_openvino_model"
            folder.mkdir()
            ov.save_model(quantized, folder / "model.xml", compress_to_fp16=False)
            (folder / "metadata.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
            models["openvino_int8"] = folder.name
        versions = {name: importlib.metadata.version(name)
                    for name in ("torch", "ultralytics", "onnx", "openvino")}
        if int8:
            versions["nncf"] = importlib.metadata.version("nncf")
        for model_format, relative in models.items():
            if model_format == "onnx":
                continue
            folder = stage / relative
            local_manifest = {
                "schema_version": 1, "task": task, "image_size": image_size,
                "source_sha256": source_digest, "precision": model_format,
                "files": {p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file()},
            }
            (folder / "manifest.json").write_text(json.dumps(local_manifest, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "schema_version": 1, "task": task, "image_size": image_size,
            "input_shape": [1, 3, image_size, image_size], "source": str(source.relative_to(ROOT)),
            "source_sha256": source_digest, "models": models, "versions": versions,
            "preprocessing": "square letterbox, BGR->RGB, NCHW float32 /255, rect=False",
            "fp16_note": "FP16 weight compression; runtime execution precision is device-dependent.",
            "int8_note": "Mixed precision PTQ; calibration is not an accuracy evaluation.",
            "calibration": calibration,
            "files": {str(p.relative_to(stage)).replace("\\", "/"): sha256(p)
                      for p in sorted(stage.rglob("*")) if p.is_file()},
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        stage.rename(output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("pose", "detect"), default="pose")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--calibration-images", type=Path)
    parser.add_argument("--calibration-limit", type=int, default=256)
    parser.add_argument("--weights", type=Path,
                        help="Detect-only explicit checkpoint; requires --source-sha256.")
    parser.add_argument("--source-sha256",
                        help="Pinned SHA256 for --weights; never inferred from the file itself.")
    args = parser.parse_args()
    manifest = export_bundle(args.task, args.output, image_size=args.size, threads=args.threads,
                             int8=args.int8, calibration_images=args.calibration_images,
                             calibration_limit=args.calibration_limit,
                             weights=args.weights, source_sha256=args.source_sha256)
    print(json.dumps({"output": str(args.output.resolve()), "models": manifest["models"]}, indent=2))


if __name__ == "__main__":
    main()