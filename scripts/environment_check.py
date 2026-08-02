from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinytrantrum.config import RunConfig
from tinytrantrum.data import ensure_dataset, sha256, vocabulary
from tinytrantrum.environment import report


def main() -> None:
    config = RunConfig()
    path = ensure_dataset(config)
    text = path.read_text(encoding="utf-8")
    chars, _, _ = vocabulary(text)
    result = {
        "environment": report(),
        "dataset": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "characters": len(chars),
        "torch_random_check": float(torch.rand(1).item()),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
