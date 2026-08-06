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
from tinytrantrum.data import ensure_dataset
from tinytrantrum.interpretability import attention_for_text
from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.tokenizer import CharacterTokenizer
from tinytrantrum.training import TrainingConfig, build_optimizer


DEFAULT_PROMPT = "ROMEO:\nBut soft, what light through yonder window breaks?\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a measurable attention interpretability report")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/full_run_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/attention"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--local-window", type=int, default=4)
    parser.add_argument("--long-range-distance", type=int, default=8)
    return parser.parse_args()


def attention_stats(matrix: torch.Tensor, local_window: int, long_range_distance: int) -> dict[str, float]:
    length = matrix.shape[0]
    rows = torch.arange(length).unsqueeze(1)
    columns = torch.arange(length).unsqueeze(0)
    distance = rows - columns
    valid_previous = distance == 1
    valid_local = (distance >= 1) & (distance <= local_window)
    valid_long_range = distance >= long_range_distance
    return {
        "self_mass": float(matrix.diag().mean()),
        "previous_token_mass": float(matrix.masked_select(valid_previous).mean()) if length > 1 else 0.0,
        "local_window_mass": float(matrix.masked_select(valid_local).sum() / length),
        "long_range_mass": float(matrix.masked_select(valid_long_range).sum() / length),
        "future_mass": float(matrix.triu(diagonal=1).sum()),
    }


def save_heatmap(
    matrix: torch.Tensor,
    labels: list[str],
    output: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix.numpy(), cmap="magma", aspect="auto", vmin=0.0)
    axis.set_title(title)
    if len(labels) <= 64:
        axis.set_xticks(range(len(labels)), labels, rotation=90)
        axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Keys attended to")
    axis.set_ylabel("Query positions")
    figure.colorbar(image, ax=axis, label="attention weight")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        fallback = Path("checkpoints/full_run_best.pt")
        if fallback.exists():
            args.checkpoint = fallback
        else:
            raise FileNotFoundError(args.checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus = ensure_dataset(RunConfig()).read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(corpus)
    model = CharacterTransformer(ModelConfig(vocabulary_size=tokenizer.vocabulary_size))
    optimizer = build_optimizer(model, TrainingConfig())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_checkpoint(args.checkpoint, model, optimizer, device=device)
    model.to(device).eval()

    labels, maps = attention_for_text(model, tokenizer, args.prompt, device)
    candidates: list[dict[str, object]] = []
    for layer, layer_maps in enumerate(maps):
        for head, matrix in enumerate(layer_maps):
            stats = attention_stats(matrix, args.local_window, args.long_range_distance)
            candidates.append({"layer": layer, "head": head, **stats})

    selections = {
        "previous_token": max(candidates, key=lambda row: row["previous_token_mass"]),
        "local_window": max(candidates, key=lambda row: row["local_window_mass"]),
        "long_range": max(candidates, key=lambda row: row["long_range_mass"]),
    }
    heatmaps: dict[str, str] = {}
    for name, selection in selections.items():
        layer = int(selection["layer"])
        head = int(selection["head"])
        filename = f"{name}_layer{layer}_head{head}.png"
        save_heatmap(
            maps[layer][head],
            labels,
            args.output_dir / filename,
            f"{name.replace('_', ' ').title()} · layer {layer}, head {head}",
        )
        heatmaps[name] = filename

    summary = {
        "checkpoint": str(args.checkpoint.as_posix()),
        "prompt": args.prompt,
        "device": str(device),
        "layers": len(maps),
        "heads_per_layer": len(maps[0]),
        "tokens": len(labels),
        "local_window": args.local_window,
        "long_range_distance": args.long_range_distance,
        "causal_future_mass_max": max(row["future_mass"] for row in candidates),
        "head_statistics": candidates,
        "selected_heads": selections,
        "heatmaps": heatmaps,
    }
    (args.output_dir / "attention_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    report = f"""# TinyTantrum attention interpretability report

Checkpoint: `{args.checkpoint.as_posix()}`
Prompt: `{args.prompt.replace(chr(10), ' / ')}`
Device used for extraction: `{device}`

## What was measured

Every layer/head was scored by its average self-attention mass, previous-token mass, attention inside the previous {args.local_window} positions, long-range mass at distances of at least {args.long_range_distance}, and forbidden future-token mass.

The maximum future-token mass across all heads was `{summary['causal_future_mass_max']:.8f}`. This is the runtime evidence that the extracted maps respect the causal mask.

## Selected heads

| Selection rule | Layer | Head | Previous-token mass | Local-window mass | Long-range mass | Self mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for name, selection in selections.items():
        report += (
            f"| Highest {name.replace('_', ' ')} | {selection['layer']} | {selection['head']} | "
            f"{selection['previous_token_mass']:.4f} | {selection['local_window_mass']:.4f} | "
            f"{selection['long_range_mass']:.4f} | {selection['self_mass']:.4f} |\n"
        )
    report += """
## Interpretation limits

The selected heads are chosen by transparent statistics rather than visual cherry-picking. High previous-token or local-window mass is consistent with short-range routing; high long-range mass shows that a head uses distant context. These measurements do not prove that a head has a human-interpretable linguistic function. The report therefore treats them as behavioral evidence, not semantic claims.

Heatmaps:

"""
    for name, filename in heatmaps.items():
        report += f"- [{name.replace('_', ' ').title()} heatmap]({filename})\n"
    (args.output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
