from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from .model import CharacterTransformer
from .checkpointing import load_checkpoint, save_checkpoint


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 64
    context_length: int = 256
    learning_rate: float = 1e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.99
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    decay_steps: int = 5_000
    validation_interval: int = 250
    validation_batches: int = 20
    seed: int = 1337
    log_interval: int = 50


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate(step: int, config: TrainingConfig) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / (config.warmup_steps + 1)
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


def build_optimizer(model: CharacterTransformer, config: TrainingConfig) -> torch.optim.Optimizer:
    """Match nanoGPT's AdamW grouping: decay matrix weights, not biases/norms."""
    decay_parameters = []
    no_decay_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay_parameters if parameter.ndim >= 2 else no_decay_parameters).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": config.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
    )


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
    optimizer = build_optimizer(model, config)
    generator = torch.Generator().manual_seed(config.seed)
    history: list[dict[str, float]] = []
    model.train()
    for step in range(steps):
        inputs, targets = sample_batch(train_data, config, device, generator)
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.param_groups[0]["lr"] = learning_rate(step, config)
        optimizer.step()
        if step == 0 or (step + 1) % config.validation_interval == 0 or step == steps - 1:
            metrics = estimate_loss(model, train_data, validation_data, config, device)
            metrics["step"] = float(step + 1)
            metrics["learning_rate"] = learning_rate(step, config)
            history.append(metrics)
    return history


def train_resumable(
    model: CharacterTransformer,
    train_data: Tensor,
    validation_data: Tensor,
    total_steps: int,
    config: TrainingConfig,
    checkpoint_path: Path,
    *,
    checkpoint_interval: int = 100,
    device: torch.device | None = None,
    resume: bool = False,
    progress_callback: Callable[[dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    """Train to total_steps, optionally restoring all state from a checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = build_optimizer(model, config)
    generator = torch.Generator().manual_seed(config.seed)
    start_step = 0
    best_validation_loss = None
    if resume:
        restored = load_checkpoint(checkpoint_path, model, optimizer, device=device, data_generator=generator)
        start_step = restored["step"]
        best_validation_loss = restored["best_validation_loss"]
        if start_step > total_steps:
            raise ValueError("Checkpoint step is greater than total_steps")
    history: list[dict[str, float]] = []
    started_at = time.perf_counter()
    model.train()
    for step in range(start_step, total_steps):
        inputs, targets = sample_batch(train_data, config, device, generator)
        _, loss = model(inputs, targets)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.gradient_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        current_lr = learning_rate(step, config)
        optimizer.param_groups[0]["lr"] = current_lr
        optimizer.step()
        current_step = step + 1
        progress = {
            "step": float(current_step),
            "train_batch_loss": float(loss.item()),
            "learning_rate": current_lr,
            "elapsed_seconds": time.perf_counter() - started_at,
        }
        if current_step == total_steps or current_step % config.validation_interval == 0:
            metrics = estimate_loss(model, train_data, validation_data, config, device)
            metrics["step"] = float(current_step)
            metrics["learning_rate"] = current_lr
            history.append(metrics)
            progress.update({"train_loss": metrics["train"], "validation_loss": metrics["validation"]})
            if best_validation_loss is None or metrics["validation"] < best_validation_loss:
                best_validation_loss = metrics["validation"]
        if progress_callback and (current_step % config.log_interval == 0 or current_step == total_steps):
            progress_callback(progress)
        if current_step % checkpoint_interval == 0 or current_step == total_steps:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step=current_step,
                best_validation_loss=best_validation_loss,
                metadata={"training_config": config.__dict__},
                data_generator=generator,
            )
    return history
