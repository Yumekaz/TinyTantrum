from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CharacterTokenizer:
    """Deterministic character-level tokenizer with no external vocabulary."""

    chars: tuple[str, ...]

    @classmethod
    def from_text(cls, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("Cannot build a tokenizer from empty text")
        return cls(tuple(sorted(set(text))))

    def __post_init__(self) -> None:
        if not self.chars or tuple(sorted(set(self.chars))) != self.chars:
            raise ValueError("chars must be a non-empty sorted tuple of unique characters")

    @property
    def vocabulary_size(self) -> int:
        return len(self.chars)

    @property
    def stoi(self) -> dict[str, int]:
        return {char: index for index, char in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        mapping = self.stoi
        unknown = sorted({char for char in text if char not in mapping})
        if unknown:
            raise ValueError(f"Text contains characters outside the vocabulary: {unknown!r}")
        return [mapping[char] for char in text]

    def decode(self, tokens: list[int] | tuple[int, ...]) -> str:
        decoded: list[str] = []
        for token in tokens:
            if not isinstance(token, int) or not 0 <= token < self.vocabulary_size:
                raise ValueError(f"Token {token!r} is outside the vocabulary")
            decoded.append(self.chars[token])
        return "".join(decoded)
