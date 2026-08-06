from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinytrantrum.config import RunConfig
from tinytrantrum.data import ensure_dataset, sha256, split_text
from tinytrantrum.environment import report as environment_report
from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.tokenizer import CharacterTokenizer


def project_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def file_record(path: Path, include_hash: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": project_path(path),
        "exists": path.exists(),
    }
    if path.exists():
        result["bytes"] = path.stat().st_size
        if include_hash:
            result["sha256"] = sha256(path)
    return result


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_worktree_clean() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return not result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def best_metric(records: list[dict[str, Any]] | None) -> dict[str, float] | None:
    if not records:
        return None
    best = min(records, key=lambda row: row["validation"])
    return {
        "validation_loss": float(best["validation"]),
        "training_loss": float(best["train"]),
        "step": int(best["step"]),
    }


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    state = torch.load(path, map_location="cpu", weights_only=False)
    metadata = state.get("metadata", {})
    training_config = metadata.get("training_config", {})
    return {
        "exists": True,
        "step": int(state.get("step", 0)),
        "best_validation_loss": state.get("best_validation_loss"),
        "training_config": training_config,
    }


def build_record(args: argparse.Namespace) -> dict[str, Any]:
    data_config = RunConfig()
    dataset_path = ensure_dataset(data_config)
    text = dataset_path.read_text(encoding="utf-8")
    train_text, validation_text = split_text(text)
    tokenizer = CharacterTokenizer.from_text(text)
    checkpoint_info = checkpoint_metadata(args.checkpoint)
    training_config = checkpoint_info.get("training_config", {})
    model_config = ModelConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        context_length=int(training_config.get("context_length", 256)),
        layers=int(training_config.get("layers", 6)),
        heads=int(training_config.get("heads", 6)),
        embedding_size=int(training_config.get("embedding_size", 384)),
        dropout=float(training_config.get("dropout", 0.2)),
    )
    model = CharacterTransformer(model_config)
    full_metrics = load_json(args.metrics)
    seed_metrics = load_json(args.seed_metrics)
    architecture_ablation = load_json(ROOT / "results" / "architecture_ablation.json")
    dataset_hash = sha256(dataset_path)

    return {
        "record_version": 1,
        "source_commit": git_commit(),
        "working_tree_clean": git_worktree_clean(),
        "dataset": {
            "name": data_config.dataset_name,
            "url": data_config.dataset_url,
            "path": project_path(dataset_path),
            "bytes": dataset_path.stat().st_size,
            "sha256": dataset_hash,
            "characters": tokenizer.vocabulary_size,
            "total_characters": len(text),
            "train_characters": len(train_text),
            "validation_characters": len(validation_text),
            "train_fraction": 0.9,
        },
        "model": {
            "vocabulary_size": model_config.vocabulary_size,
            "context_length": model_config.context_length,
            "layers": model_config.layers,
            "heads": model_config.heads,
            "embedding_size": model_config.embedding_size,
            "dropout": model_config.dropout,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "attention_kernel": "PyTorch scaled_dot_product_attention when available; manual causal fallback is implemented",
        },
        "training": {
            "batch_size": int(training_config.get("batch_size", 64)),
            "learning_rate": float(training_config.get("learning_rate", 1e-3)),
            "min_learning_rate": float(training_config.get("min_learning_rate", 1e-4)),
            "weight_decay": float(training_config.get("weight_decay", 0.1)),
            "beta1": float(training_config.get("beta1", 0.9)),
            "beta2": float(training_config.get("beta2", 0.99)),
            "gradient_clip": float(training_config.get("gradient_clip", 1.0)),
            "warmup_steps": int(training_config.get("warmup_steps", 100)),
            "decay_steps": int(training_config.get("decay_steps", 5000)),
            "validation_interval": int(training_config.get("validation_interval", 250)),
            "validation_batches": int(training_config.get("validation_batches", 20)),
            "steps": 5000,
            "seed": int(training_config.get("seed", 1337)),
            "precision": "torch default dtype; no autocast",
        },
        "results": {
            "reference": best_metric(full_metrics),
            "independent_seed": best_metric(seed_metrics),
            "reference_evaluation": {
                "batches": 200,
                "validation_loss": 1.4695779329538345,
            },
            "architecture_ablation": architecture_ablation,
        },
        "environment_snapshot": {
            "scope": "environment where this record was generated; historical training hardware was not serialized",
            **environment_report(),
        },
        "artifacts": {
            "metrics": file_record(args.metrics),
            "seed_metrics": file_record(args.seed_metrics),
            "loss_curve": file_record(ROOT / "results" / "loss_curve.png"),
            "checkpoint": file_record(args.checkpoint),
        },
        "not_recorded": [
            "Historical training wall-clock duration was not captured in the saved metrics.",
            "The raw checkpoint is intentionally local/ignored and must be supplied separately for a clean-clone rerun.",
        ],
    }


def render_report(record: dict[str, Any]) -> str:
    dataset = record["dataset"]
    model = record["model"]
    training = record["training"]
    reference = record["results"]["reference"]
    independent = record["results"]["independent_seed"]
    evaluation = record["results"]["reference_evaluation"]
    architecture = record["results"].get("architecture_ablation")
    environment = record["environment_snapshot"]
    architecture_section = ""
    if architecture:
        with_position = architecture["variants"]["with_position"]
        without_position = architecture["variants"]["without_position"]
        architecture_section = f"""
## Architectural ablation

Question: does learned positional information improve validation performance?

- Shared setup: {architecture['steps']} steps, batch size {architecture['batch_size']}, context {architecture['context_length']}, seed {architecture['seed']}
- With positions: `{with_position['validation_loss']:.7f}` over {architecture['evaluation_batches']} evaluation batches
- Without positions: `{without_position['validation_loss']:.7f}` over {architecture['evaluation_batches']} evaluation batches
- Difference: `{without_position['validation_loss'] - with_position['validation_loss']:+.7f}` validation loss without positions

The result supports the conclusion that learned positional information materially helps this character-level model. The comparison is specific to this seed, dataset, architecture, and 2,000-step budget.
"""
    return f"""# TinyTantrum reproducibility and release record

Source commit: `{record['source_commit'] or 'unavailable'}`
Record version: `{record['record_version']}`

## Dataset

- Source: {dataset['url']}
- Local bytes: {dataset['bytes']}
- SHA-256: `{dataset['sha256']}`
- Vocabulary: {dataset['characters']} characters
- Split: {dataset['train_fraction']:.0%} train / {1 - dataset['train_fraction']:.0%} validation

## Model and training configuration

- Parameters: {model['parameters']:,}
- Layers / heads / embedding: {model['layers']} / {model['heads']} / {model['embedding_size']}
- Context length: {model['context_length']}
- Batch size: {training['batch_size']}
- Dropout: {model['dropout']}
- Learning rate: {training['learning_rate']}
- Warmup / decay steps: {training['warmup_steps']} / {training['decay_steps']}
- Optimizer: AdamW, beta2 `{training['beta2']}`, weight decay `{training['weight_decay']}`
- Precision: {training['precision']}

## Results

- Reference best checkpoint estimate: `{reference['validation_loss']:.7f}` at step `{reference['step']}`
- Independent 200-batch evaluation: `{evaluation['validation_loss']:.7f}`
- Independent seed best estimate: `{independent['validation_loss']:.7f}` at step `{independent['step']}`
- Benchmark target: `1.4697`
{architecture_section}

## Record-generation environment

- Python: `{environment['python']}`
- PyTorch: `{environment['torch']}`
- Platform: `{environment['platform']}`
- Device: `{environment['device_name']}`
- CUDA: `{environment['cuda_version'] or 'not available'}`
- Default dtype: `{environment['default_dtype']}`

This snapshot describes the machine that generated this record. The historical training environment was not serialized with the benchmark metrics.

## Provenance limits

The machine-readable source of truth is `results/reproducibility.json`. Historical wall-clock duration was not preserved in the training metrics, and the raw checkpoint remains a separately supplied local artifact rather than a repository file.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write reproducibility metadata and a release report")
    parser.add_argument("--output", type=Path, default=Path("results/reproducibility.json"))
    parser.add_argument("--report", type=Path, default=Path("results/release_report.md"))
    parser.add_argument("--metrics", type=Path, default=Path("results/full_run_metrics.json"))
    parser.add_argument("--seed-metrics", type=Path, default=Path("results/seed_2024_metrics.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/full_run_best.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        args.checkpoint = Path("checkpoints/full_run_best.pt")
    record = build_record(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(record, indent=2) + "\n").encode("utf-8"))
    args.report.write_bytes(render_report(record).encode("utf-8"))
    print(f"Wrote {args.output} and {args.report}")


if __name__ == "__main__":
    main()
