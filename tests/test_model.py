from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import torch

from tinytrantrum.model import CharacterTransformer, ModelConfig


@pytest.fixture
def small_model() -> CharacterTransformer:
    torch.manual_seed(7)
    return CharacterTransformer(
        ModelConfig(vocabulary_size=11, context_length=16, layers=2, heads=2, embedding_size=16, dropout=0.0)
    )


def test_forward_shapes_and_loss(small_model: CharacterTransformer) -> None:
    tokens = torch.randint(0, 11, (3, 8))
    logits, loss = small_model(tokens, tokens)
    assert logits.shape == (3, 8, 11)
    assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)


def test_gradients_flow_through_core_model(small_model: CharacterTransformer) -> None:
    tokens = torch.randint(0, 11, (2, 6))
    _, loss = small_model(tokens, tokens)
    assert loss is not None
    loss.backward()
    assert small_model.token_embedding.weight.grad is not None
    assert torch.isfinite(small_model.token_embedding.weight.grad).all()


def test_context_length_is_enforced(small_model: CharacterTransformer) -> None:
    with pytest.raises(ValueError, match="context_length"):
        small_model(torch.zeros((1, 17), dtype=torch.long))
