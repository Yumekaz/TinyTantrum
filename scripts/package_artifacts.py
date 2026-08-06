from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package local TinyTantrum checkpoints for release")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("artifacts/checkpoints"))
    parser.add_argument("--output", type=Path, default=Path("dist/tinytrantrum-checkpoints.zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = ROOT / checkpoint_dir
    output = args.output
    if not output.is_absolute():
        output = ROOT / output

    checkpoints = sorted(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        raise SystemExit(f"No .pt checkpoints found in {checkpoint_dir}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": checkpoint_dir.relative_to(ROOT).as_posix(),
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in checkpoints
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        for path in checkpoints:
            archive.write(path, arcname=f"artifacts/checkpoints/{path.name}")

    print(f"Packaged {len(checkpoints)} checkpoints into {output}")
    for entry in manifest["files"]:
        print(f"- {entry['name']}: {entry['bytes']} bytes | sha256={entry['sha256']}")


if __name__ == "__main__":
    main()
