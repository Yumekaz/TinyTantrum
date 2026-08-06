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
from tinytrantrum.generation import generate
from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.tokenizer import CharacterTokenizer
from tinytrantrum.training import TrainingConfig, build_optimizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a TinyTantrum checkpoint")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/checkpoints/full_run_best.pt"))
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = ensure_dataset(RunConfig()).read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(text)
    model = CharacterTransformer(ModelConfig(vocabulary_size=tokenizer.vocabulary_size))
    optimizer = build_optimizer(model, TrainingConfig())
    load_checkpoint(args.checkpoint, model, optimizer, device=device)
    model.to(device)
    prompt = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
    output = generate(model, prompt, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
