from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch

from tinytrantrum.checkpointing import load_checkpoint, save_checkpoint
from tinytrantrum.model import CharacterTransformer, ModelConfig


def build_model() -> CharacterTransformer:
    return CharacterTransformer(
        ModelConfig(vocabulary_size=7, context_length=8, layers=1, heads=1, embedding_size=8, dropout=0.0)
    )


def test_checkpoint_restores_model_optimizer_and_metadata(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    tokens = torch.randint(0, 7, (2, 6))
    _, loss = model(tokens, tokens)
    assert loss is not None
    loss.backward()
    optimizer.step()
    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_optimizer = optimizer.state_dict()

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=17, best_validation_loss=1.23, metadata={"run": "test"})

    restored_model = build_model()
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.5)
    result = load_checkpoint(path, restored_model, restored_optimizer)

    assert result == {"step": 17, "best_validation_loss": 1.23, "metadata": {"run": "test"}}
    for name, value in restored_model.state_dict().items():
        assert torch.equal(value, expected_model[name])
    restored_state = restored_optimizer.state_dict()
    assert restored_state["param_groups"] == expected_optimizer["param_groups"]
    assert restored_state["state"].keys() == expected_optimizer["state"].keys()
    for parameter_id, expected_state in expected_optimizer["state"].items():
        for name, expected_value in expected_state.items():
            actual_value = restored_state["state"][parameter_id][name]
            if torch.is_tensor(expected_value):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def test_checkpoint_replace_is_atomic(tmp_path: Path) -> None:
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "nested" / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, step=0, best_validation_loss=None)
    assert path.exists()
    assert not path.with_suffix(".pt.tmp").exists()
