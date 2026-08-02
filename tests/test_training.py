from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest
import torch

from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.training import TrainingConfig, build_optimizer, learning_rate, sample_batch, train_steps


def test_learning_rate_warms_and_decays() -> None:
    config = TrainingConfig(learning_rate=1.0, min_learning_rate=0.1, warmup_steps=2, decay_steps=6)
    assert learning_rate(0, config) == pytest.approx(1.0 / 3.0)
    assert learning_rate(2, config) == pytest.approx(1.0)
    assert learning_rate(6, config) == pytest.approx(0.1)


def test_sample_batch_has_next_token_targets() -> None:
    config = TrainingConfig(batch_size=3, context_length=4)
    generator = torch.Generator().manual_seed(3)
    inputs, targets = sample_batch(torch.arange(20), config, torch.device("cpu"), generator)
    assert inputs.shape == targets.shape == (3, 4)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])


def test_tiny_model_can_train_without_nan() -> None:
    torch.manual_seed(3)
    model = CharacterTransformer(ModelConfig(vocabulary_size=8, context_length=8, layers=1, heads=2, embedding_size=8, dropout=0.0))
    data = torch.tensor([i % 8 for i in range(160)], dtype=torch.long)
    config = TrainingConfig(batch_size=4, context_length=8, validation_interval=4, validation_batches=2, warmup_steps=2, decay_steps=12, seed=3)
    history = train_steps(model, data, data, steps=12, config=config, device=torch.device("cpu"))
    assert history
    assert all(torch.isfinite(torch.tensor(row["train"] + row["validation"])) for row in history)
    assert history[-1]["train"] < history[0]["train"]


def test_optimizer_matches_decay_grouping() -> None:
    model = CharacterTransformer(ModelConfig(vocabulary_size=8, context_length=8, layers=1, heads=2, embedding_size=8, dropout=0.0))
    optimizer = build_optimizer(model, TrainingConfig(weight_decay=0.1))
    assert sorted(group["weight_decay"] for group in optimizer.param_groups) == [0.0, 0.1]
