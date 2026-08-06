from __future__ import annotations

from pathlib import Path

import torch

from .model import CharacterTransformer
from .tokenizer import CharacterTokenizer


@torch.no_grad()
def attention_for_text(
    model: CharacterTransformer,
    tokenizer: CharacterTokenizer,
    text: str,
    device: torch.device,
) -> tuple[list[str], list[torch.Tensor]]:
    tokens = tokenizer.encode(text)[-model.config.context_length :]
    inputs = torch.tensor([tokens], dtype=torch.long, device=device)
    _, _, attention_maps = model(inputs, return_attention=True)
    labels = list(tokenizer.decode(tokens))
    return labels, [attention[0].detach().cpu() for attention in attention_maps]


def save_attention_heatmap(
    model: CharacterTransformer,
    tokenizer: CharacterTokenizer,
    text: str,
    output: Path,
    *,
    layer: int = -1,
    head: int = 0,
    device: torch.device | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels, attention_maps = attention_for_text(model.to(device).eval(), tokenizer, text, device)
    if not -len(attention_maps) <= layer < len(attention_maps):
        raise ValueError("layer is outside the model depth")
    selected = attention_maps[layer]
    if not 0 <= head < selected.shape[0]:
        raise ValueError("head is outside the model head count")
    matrix = selected[head].numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="magma", aspect="auto")
    axis.set_title(f"Attention: layer {layer % len(attention_maps)}, head {head}")
    axis.set_xlabel("Keys attended to")
    axis.set_ylabel("Query positions")
    if len(labels) <= 64:
        axis.set_xticks(range(len(labels)), labels, rotation=90)
        axis.set_yticks(range(len(labels)), labels)
    figure.colorbar(image, ax=axis, label="attention weight")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output
