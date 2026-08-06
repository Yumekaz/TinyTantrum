from __future__ import annotations

import torch
from torch import Tensor

from .model import CharacterTransformer


@torch.no_grad()
def generate(
    model: CharacterTransformer,
    prompt: Tensor,
    max_new_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    model.eval()
    tokens = prompt
    for _ in range(max_new_tokens):
        context = tokens[:, -model.config.context_length :]
        logits, _ = model(context)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = float("-inf")
        probabilities = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        tokens = torch.cat((tokens, next_token), dim=1)
    return tokens
