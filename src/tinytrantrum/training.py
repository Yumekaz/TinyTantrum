from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import CharacterTransformer


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    context_length: int = 256
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    warmup_steps: int = 100
    decay_steps: int = 5_000
    validation_interval: int = 250
    validation_batches: int = 20
    seed: int = 1337


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(step: int, config: TrainingConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    if step >= config.decay_steps:
        return config.min_learning_rate
    progress = (step - config.warmup_steps) / max(1, config.decay_steps - config.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + cosine * (config.learning_rate - config.min_learning_rate)


def sample_batch(data: Tensor, config: TrainingConfig, device: torch.device, generator: torch.Generator) -> tuple[Tensor, Tensor]:
    if data.ndim != 1 or data.numel() <= config.context_length:
        raise ValueError("data must be one-dimensional and longer than context_length")
    starts = torch.randint(
        0, data.numel() - config.context_length, (config.batch_size,), generator=generator
    )
    offsets = torch.arange(config.context_length)
    inputs = data[starts[:, None] + offsets]
    targets = data[starts[:, None] + offsets + 1]
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss(
    model: CharacterTransformer,
    train_data: Tensor,
    validation_data: Tensor,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    results: dict[str, float] = {}
    for name, data in (("train", train_data), ("validation", validation_data)):
        generator = torch.Generator().manual_seed(config.seed + (0 if name == "train" else 1))
        losses = []
        for _ in range(config.validation_batches):
            inputs, targets = sample_batch(data, config, device, generator)
            _, loss = model(inputs, targets)
            assert loss is not None
            losses.append(loss.item())
        results[name] = sum(losses) / len(losses)
    model.train()
    return results


def train_steps(
    model: CharacterTransformer,
    train_data: Tensor,
    validation_data: Tensor,
    steps: int,
    config: TrainingConfig,
    device: torch.device | None = None,
) -> list[dict[str, float]]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.99))
    generator = torch.Generator().manual_seed(config.seed)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(steps):
        inputs, targets = sample_batch(train_data, config, device, generator)
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.param_groups[0]["lr"] = learning_rate(step, config)
        optimizer.step()
        if step == 0 or (step + 1) % config.validation_interval == 0 or step == steps - 1:
            metrics = estimate_loss(model, train_data, validation_data, config, device)
            metrics["step"] = float(step + 1)
            metrics["learning_rate"] = learning_rate(step, config)
            history.append(metrics)
    return history
