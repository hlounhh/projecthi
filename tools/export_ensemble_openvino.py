"""Export the pinned three-member posture ensemble to verified OpenVINO bundles.

Every member checkpoint is verified against the same SHA256 pin used at
inference time before any export starts, so an export can never quietly
contain a different model. The parent ensemble.json pins each member manifest.
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_openvino import export_bundle, sha256

PRECISION_FOLDERS = {"int8": "int8_openvino_model",
                     "fp32": "fp32_openvino_model",
                     "fp16": "fp16_openvino_model"}


def verify_member_sources(members):
    """Fail before exporting if any pinned member is missing or altered."""
    verified = []
    for name, path, expected_sha256 in members:
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Missing ensemble member: {name}")
        if path.stat().st_size > 20_000_000:
            raise ValueError(f"Ensemble member exceeds the size cap: {name}")
        if sha256(path) != expected_sha256:
            raise ValueError(f"Ensemble member SHA256 does not match its pin: {name}")
        verified.append((name, path.resolve(), expected_sha256))
    return verified


def export_ensemble(output, *, image_size=416, threads=2, precision="int8",
                    calibration_images=None, calibration_limit=256, members=None):
    from ai.posture_ensemble import MEMBERS

    if precision not in PRECISION_FOLDERS:
        raise ValueError("Precision must be int8, fp32 or fp16.")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new directory to preserve existing exports.")
    verified = verify_member_sources(MEMBERS if members is None else members)
    # A single member precision keeps the bundle small; INT8 is the CPU default.
    export_precisions = ("fp32",) if precision == "int8" else (precision,)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".ensemble-export-", dir=output.parent))
    try:
        entries = []
        for name, source, source_sha256 in verified:
            member_root = stage / name
            export_bundle("detect", member_root, image_size=image_size, threads=threads,
                          int8=precision == "int8",
                          calibration_images=calibration_images if precision == "int8" else None,
                          calibration_limit=calibration_limit,
                          weights=source, source_sha256=source_sha256,
                          precisions=export_precisions)
            folder = member_root / PRECISION_FOLDERS[precision]
            manifest_path = folder / "manifest.json"
            if not manifest_path.is_file():
                raise ValueError(f"Exporter did not produce the requested precision: {name}")
            entries.append({
                "name": name,
                "source": str(source.relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": source_sha256,
                "precision": f"openvino_{precision}",
                "directory": str(folder.relative_to(stage)).replace("\\", "/"),
                "manifest_sha256": sha256(manifest_path),
            })
        versions = {package: importlib.metadata.version(package)
                    for package in ("torch", "ultralytics", "onnx", "openvino")}
        if precision == "int8":
            versions["nncf"] = importlib.metadata.version("nncf")
        manifest = {
            "schema_version": 1, "task": "detect", "image_size": image_size,
            "input_shape": [1, 3, image_size, image_size], "precision": precision,
            "members": entries, "versions": versions,
            "preprocessing": "shared square letterbox, BGR->RGB, NCHW float32 /255, rect=False",
            "int8_note": "Mixed precision PTQ; calibration is not an accuracy evaluation.",
        }
        (stage / "ensemble.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                             encoding="utf-8")
        stage.rename(output)
        return manifest
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=416)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--precision", choices=tuple(PRECISION_FOLDERS), default="int8")
    parser.add_argument("--calibration-images", type=Path)
    parser.add_argument("--calibration-limit", type=int, default=256)
    args = parser.parse_args()
    if args.precision == "int8" and args.calibration_images is None:
        parser.error("--precision int8 requires --calibration-images.")
    manifest = export_ensemble(args.output, image_size=args.size, threads=args.threads,
                               precision=args.precision,
                               calibration_images=args.calibration_images,
                               calibration_limit=args.calibration_limit)
    print(json.dumps({"output": str(args.output.resolve()),
                      "members": manifest["members"]}, indent=2))


if __name__ == "__main__":
    main()
