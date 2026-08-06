from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinytrantrum.checkpointing import load_checkpoint
from tinytrantrum.config import RunConfig
from tinytrantrum.data import ensure_dataset
from tinytrantrum.interpretability import save_attention_heatmap
from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.tokenizer import CharacterTokenizer
from tinytrantrum.training import TrainingConfig, build_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an attention heatmap")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/full_run_best.pt"))
    parser.add_argument("--prompt", default="ROMEO: The night is calm")
    parser.add_argument("--output", type=Path, default=Path("results/attention_heatmap.png"))
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--head", type=int, default=0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = ensure_dataset(RunConfig()).read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    model = CharacterTransformer(ModelConfig(vocabulary_size=tokenizer.vocabulary_size))
    optimizer = build_optimizer(model, TrainingConfig())
    load_checkpoint(args.checkpoint, model, optimizer, device=device)
    output = save_attention_heatmap(model, tokenizer, args.prompt, args.output, layer=args.layer, head=args.head, device=device)
    print(f"Saved attention heatmap to {output}")


if __name__ == "__main__":
    main()
