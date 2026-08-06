from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinytrantrum.checkpointing import load_checkpoint
from tinytrantrum.config import RunConfig
from tinytrantrum.data import ensure_dataset, split_text
from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.tokenizer import CharacterTokenizer
from tinytrantrum.training import TrainingConfig, build_optimizer, estimate_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a TinyTantrum checkpoint")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/full_run.pt"))
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--embedding-size", type=int, default=384)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-position-embedding", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = ensure_dataset(RunConfig()).read_text(encoding="utf-8")
    train_text, validation_text = split_text(text)
    tokenizer = CharacterTokenizer.from_text(text)
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    validation_data = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    model = CharacterTransformer(ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=args.context_length,
        layers=args.layers,
        heads=args.heads,
        embedding_size=args.embedding_size,
        dropout=args.dropout,
        use_position_embedding=not args.no_position_embedding,
    ))
    training_config = TrainingConfig(
        context_length=args.context_length,
        validation_batches=args.batches,
        seed=args.seed,
    )
    optimizer = build_optimizer(model, training_config)
    checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device=device)
    metrics = estimate_loss(model.to(device), train_data, validation_data, training_config, device)
    print(json.dumps({"checkpoint": checkpoint, "evaluation_batches": args.batches, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
