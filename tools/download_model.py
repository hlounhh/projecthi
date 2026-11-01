"""Download a pinned checkpoint as data only; never execute repository scripts."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "melihuzunoglu/human-fall-detection"
REVISION = "97261b03632a9715a27481b2cfdac6a78e26d2d9"
DESTINATION = ROOT / "models" / "human-fall-detection" / REVISION


def main():
    argparse.ArgumentParser(
        description="Tai checkpoint posture detector da ghim (chi du lieu, khong chay ma repo)."
    ).parse_args()
    # Each downloaded repository revision has its own new, empty directory.
    DESTINATION.mkdir(parents=True, exist_ok=False)
    metadata_url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    entry = next(item for item in metadata["siblings"] if item["rfilename"] == "best.pt")
    lfs = entry.get("lfs", {})
    expected = lfs.get("sha256")
    size = entry.get("size")
    if not expected or not size or size > 200_000_000:
        raise RuntimeError("Missing expected LFS hash or checkpoint exceeds download limit.")
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/best.pt"
    temporary = DESTINATION / "best.pt.part"
    digest = hashlib.sha256()
    received = 0
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as output:
        while chunk := response.read(1024 * 1024):
            received += len(chunk)
            if received > size:
                raise RuntimeError("Download larger than metadata size.")
            digest.update(chunk)
            output.write(chunk)
    if received != size or digest.hexdigest() != expected:
        raise RuntimeError("Checkpoint integrity check failed; file was not promoted.")
    temporary.rename(DESTINATION / "best.pt")
    record = {
        "repository": REPO, "revision": REVISION, "filename": "best.pt",
        "sha256": expected, "size_bytes": received,
        "declared_license": "AGPL-3.0", "url": url,
        "note": "Image posture detector, not a temporal fall verifier. Hash is integrity, not trust.",
    }
    (DESTINATION / "provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Downloaded {received} bytes, SHA256 verified: {expected}")
    print(f"Checkpoint saved: {DESTINATION / 'best.pt'}")
    print("No checkpoint was loaded or repository code executed.")


if __name__ == "__main__":
    main()
