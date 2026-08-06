from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tinytrantrum.checkpointing import load_checkpoint
from tinytrantrum.config import RunConfig
from tinytrantrum.data import ensure_dataset
from tinytrantrum.generation import generate
from tinytrantrum.interpretability import attention_for_text
from tinytrantrum.model import CharacterTransformer, ModelConfig
from tinytrantrum.tokenizer import CharacterTokenizer
from tinytrantrum.training import TrainingConfig, build_optimizer


CHECKPOINT_DIR = ROOT / "artifacts" / "checkpoints"
RESULTS_DIR = ROOT / "results"


def checkpoint_choices() -> list[str]:
    return sorted(path.name for path in CHECKPOINT_DIR.glob("*_best.pt"))


@lru_cache(maxsize=4)
def load_resources(checkpoint_name: str) -> tuple[CharacterTransformer, CharacterTokenizer, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = ensure_dataset(RunConfig()).read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(corpus)
    model = CharacterTransformer(ModelConfig(vocabulary_size=tokenizer.vocabulary_size))
    optimizer = build_optimizer(model, TrainingConfig())
    load_checkpoint(CHECKPOINT_DIR / checkpoint_name, model, optimizer, device=device)
    return model.to(device).eval(), tokenizer, device


def generate_text(prompt: str, tokens: int, temperature: float, top_k: int, checkpoint_name: str) -> str:
    model, tokenizer, device = load_resources(checkpoint_name)
    encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    output = generate(model, encoded, int(tokens), temperature=float(temperature), top_k=int(top_k))
    return tokenizer.decode(output[0].tolist())


def attention_plot(prompt: str, layer: int, head: int, checkpoint_name: str):
    import matplotlib.pyplot as plt

    model, tokenizer, device = load_resources(checkpoint_name)
    labels, maps = attention_for_text(model, tokenizer, prompt, device)
    selected = maps[int(layer)][int(head)].numpy()
    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(selected, cmap="magma", aspect="auto")
    axis.set_title(f"Layer {int(layer)}, head {int(head)}")
    if len(labels) <= 64:
        axis.set_xticks(range(len(labels)), labels, rotation=90)
        axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Keys attended to")
    axis.set_ylabel("Query positions")
    figure.colorbar(image, ax=axis, label="attention weight")
    figure.tight_layout()
    return figure


def loss_plot():
    import matplotlib.pyplot as plt

    path = RESULTS_DIR / "full_run_metrics.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text(encoding="utf-8"))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot([row["step"] for row in records], [row["train"] for row in records], label="train")
    axis.plot([row["step"] for row in records], [row["validation"] for row in records], label="validation")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Cross-entropy loss")
    axis.set_title("Reference training run")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure


def ablation_table() -> list[list[object]]:
    path = RESULTS_DIR / "summary.json"
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    return [[row["context_length"], row["best"]["validation"], row["best"]["step"]] for row in records]


def main() -> None:
    try:
        import gradio as gr
    except ImportError as error:
        raise SystemExit("Install dashboard dependencies with: python -m pip install -e \".[dashboard]\"") from error

    choices = checkpoint_choices()
    if not choices:
        raise SystemExit(f"No best checkpoints found in {CHECKPOINT_DIR}")
    default_checkpoint = "full_run_best.pt" if "full_run_best.pt" in choices else choices[0]

    with gr.Blocks(title="TinyTantrum") as demo:
        gr.Markdown(
            "# TinyTantrum\n"
            "### A from-scratch character-level transformer laboratory\n"
            "Benchmark: **1.46958** validation loss against the **1.4697** reference."
        )
        with gr.Tab("Generate"):
            with gr.Row():
                prompt = gr.Textbox(value="ROMEO:", label="Prompt", lines=2)
                checkpoint = gr.Dropdown(choices=choices, value=default_checkpoint, label="Checkpoint")
            with gr.Row():
                token_count = gr.Slider(20, 800, value=300, step=10, label="New characters")
                temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.05, label="Temperature")
                top_k = gr.Slider(1, 65, value=20, step=1, label="Top-k")
            generate_button = gr.Button("Generate", variant="primary")
            generated = gr.Textbox(label="Generated text", lines=18)
            generate_button.click(generate_text, [prompt, token_count, temperature, top_k, checkpoint], generated)
        with gr.Tab("Evidence"):
            gr.Markdown("## Reference training curve")
            gr.Plot(value=loss_plot())
            gr.Markdown("## Context-length ablation\nLower validation loss is better.")
            gr.Dataframe(headers=["Context length", "Best validation loss", "Best step"], value=ablation_table(), interactive=False)
        with gr.Tab("Attention"):
            attention_prompt = gr.Textbox(value="ROMEO: The night is calm", label="Text to inspect")
            with gr.Row():
                attention_checkpoint = gr.Dropdown(choices=choices, value=default_checkpoint, label="Checkpoint")
                layer = gr.Number(value=5, precision=0, label="Layer")
                head = gr.Number(value=0, precision=0, label="Head")
            attention_button = gr.Button("Inspect attention")
            attention_image = gr.Plot(label="Attention heatmap")
            attention_button.click(attention_plot, [attention_prompt, layer, head, attention_checkpoint], attention_image)
        gr.Markdown("TinyTantrum is a small character-level model; generated text is qualitative evidence, not the benchmark.")

    demo.launch()


if __name__ == "__main__":
    main()
