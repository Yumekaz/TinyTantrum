from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch

from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.training import TrainingConfig, train_resumable


def make_model() -> CharacterTransformer:
    return CharacterTransformer(
        ModelConfig(vocabulary_size=8, context_length=8, layers=1, heads=2, embedding_size=8, dropout=0.0)
    )


def test_interrupted_and_resumed_run_matches_uninterrupted_run(tmp_path: Path) -> None:
    data = torch.tensor([i % 8 for i in range(256)], dtype=torch.long)
    config = TrainingConfig(
        batch_size=4, context_length=8, validation_interval=4, validation_batches=2,
        warmup_steps=2, decay_steps=20, seed=22,
    )
    torch.manual_seed(22)
    uninterrupted = make_model()
    train_resumable(uninterrupted, data, data, 12, config, tmp_path / "unused.pt", checkpoint_interval=6)

    torch.manual_seed(22)
    interrupted = make_model()
    checkpoint = tmp_path / "resume.pt"
    train_resumable(interrupted, data, data, 6, config, checkpoint, checkpoint_interval=6)
    train_resumable(interrupted, data, data, 12, config, checkpoint, checkpoint_interval=6, resume=True)

    for name, expected in uninterrupted.state_dict().items():
        assert torch.equal(expected, interrupted.state_dict()[name]), name
