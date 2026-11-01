"""Fetch pinned OpenMMLab ONNX assets as data, with archive and model hashes."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

BASE_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
DESTINATION = Path(__file__).resolve().parents[1] / "models" / "rtmpose-s-body7-20230504"
ASSETS = (
    ("yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", "yolox-tiny.onnx",
     "36e09ca555916253fa1b5d51bec01e48e98e1f0af966c51ce3e844d8c8fc4cfc",
     "ceb11c07298f95c50d7c5abeb906d03340c85f23aa79e3e66966e7fb6c307250"),
    ("rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip", "rtmpose-s.onnx",
     "7673922e531014906ca4f0f239b7e233b740146a10b632deaa2a28d45470d802",
     "9aeb635b83f86aea45cf45d85798f7eba1a162de8e0d721c44e54fe5eebaf47d"),
)


def download(destination=DESTINATION):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for archive_name, model_name, archive_hash, model_hash in ASSETS:
        target = destination / model_name
        if target.exists():
            if target.stat().st_size > 50_000_000 or hashlib.sha256(target.read_bytes()).hexdigest() != model_hash:
                raise ValueError(f"Existing model has an unexpected hash: {target}")
        else:
            data = bytearray()
            with urllib.request.urlopen(BASE_URL + archive_name, timeout=120) as response:
                while chunk := response.read(1024 * 1024):
                    data.extend(chunk)
                    if len(data) > 100_000_000:
                        raise ValueError("RTMPose archive exceeds download size limit.")
            if hashlib.sha256(data).hexdigest() != archive_hash:
                raise ValueError(f"Archive SHA256 mismatch: {archive_name}")
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                models = [info for info in archive.infolist() if info.filename.endswith(".onnx")]
                if len(models) != 1 or models[0].file_size > 50_000_000:
                    raise ValueError("Unexpected RTMPose archive contents.")
                # Read a member directly; never extract archive paths to the filesystem.
                model = archive.read(models[0])
            if hashlib.sha256(model).hexdigest() != model_hash:
                raise ValueError(f"Model SHA256 mismatch: {model_name}")
            temporary = target.with_suffix(".onnx.part")
            created = False
            try:
                with temporary.open("xb") as output:
                    created = True
                    output.write(model)
                temporary.replace(target)
            finally:
                if created:
                    temporary.unlink(missing_ok=True)
        records.append({"url": BASE_URL + archive_name, "archive_sha256": archive_hash,
                        "file": model_name, "sha256": model_hash,
                        "size_bytes": target.stat().st_size})
    provenance = {
        "models": records,
        "source": "OpenMMLab MMPose RTMPose, rtmlib lightweight body preset",
        "license": "OpenMMLab MMPose code: Apache-2.0; consult upstream model and training dataset terms.",
        "note": "SHA256 pinned from downloaded official assets; no independently authenticated publisher signature. "
                "HumanArt/body7 training does not establish accuracy on this application's videos.",
    }
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    arguments = parser.parse_args()
    print(json.dumps(download(arguments.destination), indent=2))


if __name__ == "__main__":
    main()