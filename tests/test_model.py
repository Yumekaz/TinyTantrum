from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import torch

from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.generation import generate


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


def test_token_embedding_and_output_head_are_tied(small_model: CharacterTransformer) -> None:
    assert small_model.token_embedding.weight.data_ptr() == small_model.lm_head.weight.data_ptr()


def test_manual_attention_fallback_matches_forward() -> None:
    config = ModelConfig(vocabulary_size=11, context_length=8, layers=1, heads=1, embedding_size=8, dropout=0.0, use_flash_attention=False)
    model = CharacterTransformer(config)
    tokens = torch.randint(0, 11, (1, 5))
    logits, loss = model(tokens, tokens)
    assert logits.shape == (1, 5, 11)
    assert loss is not None and torch.isfinite(loss)


def test_generation_preserves_prompt_and_length() -> None:
    model = CharacterTransformer(ModelConfig(vocabulary_size=5, context_length=4, layers=1, heads=1, embedding_size=4, dropout=0.0))
    prompt = torch.tensor([[1, 2]])
    output = generate(model, prompt, max_new_tokens=3, temperature=1.0)
    assert output.shape == (1, 5)
    assert torch.equal(output[:, :2], prompt)
