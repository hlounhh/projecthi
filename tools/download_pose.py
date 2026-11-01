"""Download official pose asset as data; no repository code is executed."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
DESTINATION = ROOT / "models" / "yolo11n-pose-v8.3.0"


def main():
    argparse.ArgumentParser(
        description="Tai checkpoint pose chinh thuc (chi du lieu, khong chay ma repo)."
    ).parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=False)
    path = DESTINATION / "yolo11n-pose.pt.part"
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(URL, timeout=120) as response, path.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > 30_000_000:
                raise RuntimeError("Pose checkpoint exceeds download size limit.")
            output.write(chunk)
            digest.update(chunk)
    path.rename(DESTINATION / "yolo11n-pose.pt")
    metadata = {"url": URL, "sha256": digest.hexdigest(), "size_bytes": total,
                "license": "AGPL-3.0 / Ultralytics licensing",
                "note": "Locally computed hash; not independently authenticated publisher signature."}
    (DESTINATION / "provenance.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
