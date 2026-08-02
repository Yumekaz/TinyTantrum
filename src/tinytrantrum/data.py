from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

from .config import RunConfig
from .tokenizer import CharacterTokenizer


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


def split_text(text: str, train_fraction: float = 0.9) -> tuple[str, str]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_index = int(len(text) * train_fraction)
    if split_index == 0 or split_index == len(text):
        raise ValueError("Text is too short for the requested split")
    return text[:split_index], text[split_index:]


def encode_corpus(text: str) -> tuple[CharacterTokenizer, list[int]]:
    tokenizer = CharacterTokenizer.from_text(text)
    return tokenizer, tokenizer.encode(text)
