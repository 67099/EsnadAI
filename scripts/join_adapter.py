"""Rebuilds the v5 adapter file from the two parts stored in this repository and checks it.

GitHub does not accept files above 100 MB without Git LFS, and the adapter weights are 160 MB, so
models/allam-7b-esnad-lora-v5/adapter_parts/ holds the file cut into two byte ranges. This script joins them back
into models/allam-7b-esnad-lora-v5/final_adapter/adapter_model.safetensors and verifies the SHA-256 of the result.

Run from the repository root:
    python scripts/join_adapter.py            # writes the file next to the other adapter files
    python scripts/join_adapter.py --check    # only verifies an existing file
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARTS_DIR = ROOT / "models" / "allam-7b-esnad-lora-v5" / "adapter_parts"
TARGET = ROOT / "models" / "allam-7b-esnad-lora-v5" / "final_adapter" / "adapter_model.safetensors"
EXPECTED_SHA256 = "93f4bc7505fd47e7fa889dab6e8438f6df8865d7883e0eea68d6a9f80ace2cc8"
EXPECTED_SIZE = 159967880


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path):
    size_ok = path.stat().st_size == EXPECTED_SIZE
    sha_ok = sha256_of(path) == EXPECTED_SHA256
    print(f"{path.name}: size {'ok' if size_ok else 'WRONG'}, sha256 {'ok' if sha_ok else 'WRONG'}")
    return size_ok and sha_ok


def main():
    if "--check" in sys.argv:
        sys.exit(0 if TARGET.exists() and verify(TARGET) else 1)

    parts = sorted(PARTS_DIR.glob("adapter_model.safetensors.part*"))
    if not parts:
        sys.exit(f"no parts found in {PARTS_DIR}")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    with open(TARGET, "wb") as out:
        for part in parts:
            out.write(part.read_bytes())
    print("joined", len(parts), "parts into", TARGET)
    if not verify(TARGET):
        TARGET.unlink()
        sys.exit("the joined file does not match the expected hash, so it was removed")


if __name__ == "__main__":
    main()
