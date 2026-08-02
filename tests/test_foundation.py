from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from tinytrantrum.data import vocabulary


def test_vocabulary_is_deterministic_and_complete() -> None:
    chars, stoi, itos = vocabulary("banana")
    assert chars == ["a", "b", "n"]
    assert set(stoi) == set(chars)
    assert {index: char for char, index in stoi.items()} == itos
