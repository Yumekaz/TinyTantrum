from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot TinyTantrum training metrics")
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/loss_curve.png"))
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Install matplotlib before plotting: python -m pip install matplotlib") from error

    records = json.loads(args.metrics.read_text(encoding="utf-8"))
    steps = [record["step"] for record in records]
    train = [record["train"] for record in records]
    validation = [record["validation"] for record in records]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(steps, train, label="train loss")
    plt.plot(steps, validation, label="validation loss")
    plt.xlabel("Training step")
    plt.ylabel("Cross-entropy loss")
    plt.title("TinyTantrum training curve")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=160)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
