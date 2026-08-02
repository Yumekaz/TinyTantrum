from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

from .config import RunConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dataset(config: RunConfig) -> Path:
    """Download the pinned corpus once and return its local path."""
    path = config.dataset_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urlopen(config.dataset_url, timeout=30) as response:
            path.write_bytes(response.read())
    if path.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {path}")
    actual_hash = sha256(path)
    if config.expected_sha256 and actual_hash != config.expected_sha256:
        raise ValueError(
            f"Dataset checksum mismatch: expected {config.expected_sha256}, got {actual_hash}"
        )
    return path


def vocabulary(text: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    return chars, stoi, itos
