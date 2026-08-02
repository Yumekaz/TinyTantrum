from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from .model import CharacterTransformer


def save_checkpoint(
    path: Path,
    model: CharacterTransformer,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    best_validation_loss: float | None,
    metadata: dict[str, Any] | None = None,
    data_generator: torch.Generator | None = None,
) -> None:
    """Save everything needed to continue a run deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "best_validation_loss": best_validation_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "data_rng": data_generator.get_state() if data_generator is not None else None,
        "metadata": metadata or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    model: CharacterTransformer,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device | str = "cpu",
    data_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if torch.cuda.is_available() and state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    if data_generator is not None and state.get("data_rng") is not None:
        data_generator.set_state(state["data_rng"])
    return {
        "step": int(state["step"]),
        "best_validation_loss": state["best_validation_loss"],
        "metadata": state.get("metadata", {}),
    }
