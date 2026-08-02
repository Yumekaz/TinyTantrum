from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from tinytrantrum.data import encode_corpus, split_text
from tinytrantrum.tokenizer import CharacterTokenizer


def test_encode_decode_round_trip() -> None:
    tokenizer = CharacterTokenizer.from_text("banana!")
    text = "!banana"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    assert tokenizer.vocabulary_size == 4


def test_tokenizer_is_deterministic() -> None:
    assert CharacterTokenizer.from_text("cba").chars == ("a", "b", "c")


def test_unknown_character_is_rejected() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tokenizer.encode("abd")


def test_invalid_token_is_rejected() -> None:
    tokenizer = CharacterTokenizer.from_text("abc")
    with pytest.raises(ValueError, match="outside the vocabulary"):
        tokenizer.decode([3])


def test_split_and_encode_corpus() -> None:
    train, validation = split_text("abcdefghij", train_fraction=0.7)
    assert (train, validation) == ("abcdefg", "hij")
    tokenizer, encoded = encode_corpus("abcabc")
    assert tokenizer.decode(encoded) == "abcabc"
