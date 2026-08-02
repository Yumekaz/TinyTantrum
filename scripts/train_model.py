from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinytrantrum.config import RunConfig
from tinytrantrum.data import ensure_dataset, split_text
from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.tokenizer import CharacterTokenizer
from tinytrantrum.training import TrainingConfig, seed_everything, train_resumable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyTantrum from scratch")
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--embedding-size", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--metrics", type=Path, default=Path("runs/metrics.json"))
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    corpus_path = ensure_dataset(RunConfig())
    text = corpus_path.read_text(encoding="utf-8")
    train_text, validation_text = split_text(text)
    tokenizer = CharacterTokenizer.from_text(text)
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    validation_data = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    model_config = ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=args.context_length,
        layers=args.layers,
        heads=args.heads,
        embedding_size=args.embedding_size,
        dropout=args.dropout,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    model = CharacterTransformer(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started_at = time.perf_counter()

    def report(progress: dict[str, float]) -> None:
        fields = [
            f"step={int(progress['step'])}/{args.steps}",
            f"batch_loss={progress['train_batch_loss']:.4f}",
        ]
        if "train_loss" in progress:
            fields.extend([
                f"train={progress['train_loss']:.4f}",
                f"validation={progress['validation_loss']:.4f}",
            ])
        fields.append(f"lr={progress['learning_rate']:.2e}")
        fields.append(f"elapsed={time.perf_counter() - started_at:.1f}s")
        print(" | ".join(fields), flush=True)

    history = train_resumable(
        model,
        train_data,
        validation_data,
        args.steps,
        training_config,
        args.checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        device=device,
        resume=args.resume,
        progress_callback=report,
    )
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(json.dumps({"device": str(device), "vocabulary_size": tokenizer.vocabulary_size, "metrics": history}, indent=2))


if __name__ == "__main__":
    main()
