from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched context-length experiments")
    parser.add_argument("--contexts", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/context_ablation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for context in args.contexts:
        checkpoint = args.output_dir / f"context_{context}.pt"
        metrics = args.output_dir / f"context_{context}.json"
        command = [
            sys.executable,
            str(root / "scripts" / "train_model.py"),
            "--steps", str(args.steps),
            "--context-length", str(context),
            "--checkpoint", str(checkpoint),
            "--metrics", str(metrics),
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=root, check=True)
        history = json.loads(metrics.read_text(encoding="utf-8"))
        best = min(history, key=lambda row: row["validation"])
        records.append({"context_length": context, "best": best})
    summary = args.output_dir / "summary.json"
    summary.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"Saved ablation summary to {summary}")


if __name__ == "__main__":
    main()
