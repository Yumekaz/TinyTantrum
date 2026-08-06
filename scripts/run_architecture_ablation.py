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
    parser = argparse.ArgumentParser(description="Run a controlled positional-embedding ablation")
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--embedding-size", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--validation-interval", type=int, default=250)
    parser.add_argument("--validation-batches", type=int, default=20)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/architecture_ablation"))
    return parser.parse_args()


def best_result(history: list[dict[str, float]]) -> dict[str, float | int]:
    best = min(history, key=lambda row: row["validation"])
    return {
        "validation": float(best["validation"]),
        "train": float(best["train"]),
        "step": int(best["step"]),
    }


def run_variant(
    *,
    name: str,
    use_position_embedding: bool,
    args: argparse.Namespace,
    train_data: torch.Tensor,
    validation_data: torch.Tensor,
    vocabulary_size: int,
    device: torch.device,
) -> dict[str, object]:
    seed_everything(args.seed)
    model_config = ModelConfig(
        vocabulary_size=vocabulary_size,
        context_length=args.context_length,
        layers=args.layers,
        heads=args.heads,
        embedding_size=args.embedding_size,
        dropout=args.dropout,
        use_position_embedding=use_position_embedding,
    )
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        context_length=args.context_length,
        seed=args.seed,
        validation_interval=args.validation_interval,
        validation_batches=args.validation_batches,
        decay_steps=args.steps,
        log_interval=args.log_interval,
    )
    model = CharacterTransformer(model_config)
    checkpoint = args.output_dir / f"{name}.pt"
    metrics_path = args.output_dir / f"{name}_metrics.json"
    prior_history: list[dict[str, float]] = []
    if args.resume and metrics_path.exists():
        prior_history = json.loads(metrics_path.read_text(encoding="utf-8"))

    def report(progress: dict[str, float]) -> None:
        fields = [
            f"[{name}] step={int(progress['step'])}/{args.steps}",
            f"batch_loss={progress['train_batch_loss']:.4f}",
        ]
        if "train_loss" in progress:
            row = {
                "train": float(progress["train_loss"]),
                "validation": float(progress["validation_loss"]),
                "step": float(progress["step"]),
                "learning_rate": float(progress["learning_rate"]),
            }
            prior_history[:] = [item for item in prior_history if item["step"] != row["step"]]
            prior_history.append(row)
            metrics_path.write_text(json.dumps(sorted(prior_history, key=lambda item: item["step"]), indent=2) + "\n", encoding="utf-8")
            fields.extend([
                f"train={progress['train_loss']:.4f}",
                f"validation={progress['validation_loss']:.4f}",
            ])
        fields.extend([
            f"lr={progress['learning_rate']:.2e}",
            f"elapsed={progress['elapsed_seconds']:.1f}s",
        ])
        print(" | ".join(fields), flush=True)

    started_at = time.perf_counter()
    history = train_resumable(
        model,
        train_data,
        validation_data,
        args.steps,
        training_config,
        checkpoint,
        checkpoint_interval=args.checkpoint_interval,
        device=device,
        resume=args.resume,
        progress_callback=report,
    )
    elapsed = time.perf_counter() - started_at
    combined_history = prior_history + history
    deduplicated_history = {
        int(row["step"]): row for row in combined_history
    }
    history = [deduplicated_history[step] for step in sorted(deduplicated_history)]
    if not history:
        raise RuntimeError(f"No metrics available for {name}; rerun without --resume or keep the metrics file beside the checkpoint")
    return {
        "variant": name,
        "use_position_embedding": use_position_embedding,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best": best_result(history),
        "elapsed_seconds": elapsed,
        "metrics_path": str(metrics_path.as_posix()),
        "checkpoint_path": str(checkpoint.as_posix()),
        "config": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "layers": args.layers,
            "heads": args.heads,
            "embedding_size": args.embedding_size,
            "dropout": args.dropout,
        },
        "history": history,
    }


def write_report(summary: dict[str, object], output: Path) -> None:
    variants = summary["variants"]
    with_position = next(row for row in variants if row["variant"] == "with_position")
    without_position = next(row for row in variants if row["variant"] == "without_position")
    delta = without_position["best"]["validation"] - with_position["best"]["validation"]
    interpretation = (
        "Removing positional embeddings improved validation loss under this budget."
        if delta < 0
        else "Learned positional embeddings improved validation loss under this budget."
        if delta > 0
        else "The variants tied under this budget."
    )
    report = f"""# Positional-embedding ablation

## Question

Does the learned positional embedding improve character-level validation performance when every other training choice is held fixed?

## Controlled setup

- Seed: `{summary['shared_config']['seed']}`
- Steps: `{summary['shared_config']['steps']}`
- Batch size: `{summary['shared_config']['batch_size']}`
- Context length: `{summary['shared_config']['context_length']}`
- Model: `{summary['shared_config']['layers']}` layers, `{summary['shared_config']['heads']}` heads, `{summary['shared_config']['embedding_size']}` dimensions
- Dropout: `{summary['shared_config']['dropout']}`
- Device: `{summary['device']}`

## Results

| Variant | Parameters | Best validation loss | Best step | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| With learned positions | {with_position['parameters']:,} | {with_position['best']['validation']:.7f} | {with_position['best']['step']} | {with_position['elapsed_seconds']:.1f} |
| Without learned positions | {without_position['parameters']:,} | {without_position['best']['validation']:.7f} | {without_position['best']['step']} | {without_position['elapsed_seconds']:.1f} |

Validation-loss delta (without positions minus with positions): `{delta:+.7f}`.

**Interpretation:** {interpretation} This conclusion is limited to the recorded seed, dataset, architecture, and fixed-step budget.
"""
    output.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = ensure_dataset(RunConfig())
    text = corpus_path.read_text(encoding="utf-8")
    train_text, validation_text = split_text(text)
    tokenizer = CharacterTokenizer.from_text(text)
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    validation_data = torch.tensor(tokenizer.encode(validation_text), dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = [
        run_variant(
            name="with_position",
            use_position_embedding=True,
            args=args,
            train_data=train_data,
            validation_data=validation_data,
            vocabulary_size=tokenizer.vocabulary_size,
            device=device,
        ),
        run_variant(
            name="without_position",
            use_position_embedding=False,
            args=args,
            train_data=train_data,
            validation_data=validation_data,
            vocabulary_size=tokenizer.vocabulary_size,
            device=device,
        ),
    ]
    for variant in variants:
        metrics_path = args.output_dir / f"{variant['variant']}_metrics.json"
        metrics_path.write_text(json.dumps(variant["history"], indent=2) + "\n", encoding="utf-8")
        del variant["history"]

    summary = {
        "question": "Does learned positional information improve character-level validation loss?",
        "device": str(device),
        "shared_config": {
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "layers": args.layers,
            "heads": args.heads,
            "embedding_size": args.embedding_size,
            "dropout": args.dropout,
        },
        "variants": variants,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_report(summary, args.output_dir / "report.md")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
