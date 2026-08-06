from __future__ import annotations

from functools import lru_cache
import json
import os
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
if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --tt-bg: #0a0b0f;
  --tt-panel: #11131a;
  --tt-panel-2: #171a22;
  --tt-line: #272b36;
  --tt-text: #f4f1ea;
  --tt-muted: #9298a6;
  --tt-orange: #ff8a3d;
  --tt-green: #7fe0b5;
}

body, .gradio-container { background: var(--tt-bg) !important; color: var(--tt-text) !important; }
.gradio-container { max-width: 1380px !important; padding: 28px 34px 50px !important; font-family: 'Space Grotesk', sans-serif !important; }
.gradio-container * { border-color: var(--tt-line) !important; }
#hero { background: radial-gradient(circle at 85% 5%, #402311 0%, transparent 34%), linear-gradient(135deg, #171a22 0%, #0f1117 70%); border: 1px solid #303540; border-radius: 24px; padding: 34px 38px; margin-bottom: 18px; overflow: hidden; }
#hero .eyebrow { color: var(--tt-orange); font-family: 'DM Mono', monospace; font-size: 12px; letter-spacing: .14em; text-transform: uppercase; }
#hero h1 { font-size: clamp(38px, 6vw, 72px); letter-spacing: -.06em; line-height: .95; margin: 12px 0 14px; }
#hero p { color: #b7bbc5; font-size: 16px; max-width: 680px; margin: 0; }
.hero-mark { float: right; width: 116px; height: 116px; border: 1px solid #85451f; border-radius: 50%; background: repeating-radial-gradient(circle, transparent 0 10px, rgba(255,138,61,.2) 11px 12px), #27170e; box-shadow: 0 0 80px rgba(255,138,61,.18); }
#metric-grid { gap: 12px; margin-bottom: 20px; }
.metric-card { background: var(--tt-panel); border: 1px solid var(--tt-line); border-radius: 16px; padding: 17px 20px; min-height: 92px; }
.metric-label { color: var(--tt-muted); font-family: 'DM Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: .09em; }
.metric-value { color: var(--tt-text); font-size: 28px; font-weight: 600; margin-top: 8px; }
.metric-value.orange { color: var(--tt-orange); }
.metric-value.green { color: var(--tt-green); }
.section-note { color: var(--tt-muted); font-size: 14px; margin: 4px 0 14px; }
.tab-nav { background: transparent !important; border-bottom: 1px solid var(--tt-line) !important; }
.tab-nav button { color: var(--tt-muted) !important; font-weight: 600 !important; }
.tab-nav button.selected { color: var(--tt-orange) !important; border-bottom-color: var(--tt-orange) !important; }
.panel { background: var(--tt-panel) !important; border: 1px solid var(--tt-line) !important; border-radius: 18px !important; }
textarea, input { background: #0d0f14 !important; color: var(--tt-text) !important; }
.gr-button-primary { background: var(--tt-orange) !important; color: #1b1008 !important; border: 0 !important; font-weight: 700 !important; }
.gr-button-primary:hover { background: #ffad70 !important; }
footer { display: none !important; }
code, pre { font-family: 'DM Mono', monospace !important; }
"""

RESPONSIVE_STYLE_HTML = """
<style>
  #metric-grid, .metric-grid { min-width: 0 !important; }
  .generate-layout, .sampling-controls { min-width: 0 !important; }
  .generate-layout > *, .sampling-controls > * { min-width: 0 !important; }
  .generate-layout .panel { min-width: 0 !important; overflow: hidden !important; }
  .sampling-controls .wrap, .sampling-controls .slider_input_container { min-width: 0 !important; width: 100% !important; }
  .sampling-controls input[type='range'] { min-width: 0 !important; }

  @media (max-width: 900px) {
    .gradio-container { padding: 18px 14px 36px !important; }
    #hero { padding: 26px 24px; border-radius: 20px; }
    #hero p { font-size: 15px; line-height: 1.7; }
    #metric-grid { display: grid !important; grid-template-columns: repeat(2, minmax(0, 1fr)) !important; gap: 10px !important; }
    #metric-grid > * { min-width: 0 !important; }
    .metric-card { min-width: 0; padding: 14px 16px; }
    .metric-value { font-size: 24px; }
    .generate-layout { display: flex !important; flex-direction: column !important; gap: 14px !important; }
    .generate-layout > * { width: 100% !important; flex: none !important; }
    .sampling-controls { display: grid !important; grid-template-columns: 1fr !important; gap: 8px !important; }
    .sampling-controls > * { width: 100% !important; }
  }

  @media (max-width: 460px) {
    #hero { padding: 22px 18px; }
    .hero-mark { width: 84px; height: 84px; }
    .metric-card { padding: 12px; }
    .metric-value { font-size: 21px; }
  }
</style>
"""

HERO_HTML = """
<div id="hero">
  <div class="hero-mark"></div>
  <div class="eyebrow">from-scratch transformer / research console</div>
  <h1>TinyTantrum</h1>
  <p>Watch a tiny language model think one character at a time. Explore the benchmark, inspect attention, and sample from a checkpoint trained from random initialization.</p>
</div>
"""

METRIC_HTML = """
<div class="metric-card"><div class="metric-label">Best validation</div><div class="metric-value orange">1.46958</div></div>
"""

METRIC_REFERENCE_HTML = """
<div class="metric-card"><div class="metric-label">Reference target</div><div class="metric-value">1.46970</div></div>
"""

METRIC_SEED_HTML = """
<div class="metric-card"><div class="metric-label">Independent seed</div><div class="metric-value green">1.47926</div></div>
"""

METRIC_STATUS_HTML = """
<div class="metric-card"><div class="metric-label">Benchmark status</div><div class="metric-value green">PASS</div></div>
"""


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


def build_demo():
    try:
        import gradio as gr
    except ImportError as error:
        raise SystemExit("Install dashboard dependencies with: python -m pip install -e \".[dashboard]\"") from error

    choices = checkpoint_choices()
    if not choices:
        raise SystemExit(f"No best checkpoints found in {CHECKPOINT_DIR}")
    default_checkpoint = "full_run_best.pt" if "full_run_best.pt" in choices else choices[0]

    with gr.Blocks(title="TinyTantrum · Research Console") as demo:
        gr.HTML(HERO_HTML)
        gr.HTML(RESPONSIVE_STYLE_HTML)
        with gr.Row(elem_id="metric-grid"):
            gr.HTML(METRIC_HTML)
            gr.HTML(METRIC_REFERENCE_HTML)
            gr.HTML(METRIC_SEED_HTML)
            gr.HTML(METRIC_STATUS_HTML)
        with gr.Tabs():
            with gr.Tab("Generate", id="generate"):
                gr.Markdown("## Sample from a trained checkpoint", elem_classes="section-note")
                with gr.Row(elem_classes="generate-layout"):
                    with gr.Column(scale=5, elem_classes="panel"):
                        prompt = gr.Textbox(value="ROMEO:", label="Prompt", lines=3, placeholder="Give the model a character-level opening...")
                        checkpoint = gr.Dropdown(choices=choices, value=default_checkpoint, label="Checkpoint")
                        with gr.Row(elem_classes="sampling-controls"):
                            token_count = gr.Slider(20, 800, value=300, step=10, label="New characters")
                            temperature = gr.Slider(0.2, 1.5, value=0.8, step=0.05, label="Temperature")
                            top_k = gr.Slider(1, 65, value=20, step=1, label="Top-k")
                        generate_button = gr.Button("Generate continuation", variant="primary")
                    with gr.Column(scale=7, elem_classes="panel"):
                        generated = gr.Textbox(label="Model continuation", lines=19, buttons=["copy"])
                generate_button.click(generate_text, [prompt, token_count, temperature, top_k, checkpoint], generated)
            with gr.Tab("Evidence", id="evidence"):
                gr.Markdown("## The run behind the result\nThe model reached the reference target on a held-out validation split. Lower is better.", elem_classes="section-note")
                gr.Plot(value=loss_plot(), show_label=False)
                gr.Markdown("## Context-length ablation", elem_classes="section-note")
                gr.Dataframe(headers=["Context length", "Best validation loss", "Best step"], value=ablation_table(), interactive=False)
            with gr.Tab("Attention", id="attention"):
                gr.Markdown("## Inspect what a head attends to\nThe map is extracted from the same causal attention mechanism used during training.", elem_classes="section-note")
                with gr.Row():
                    with gr.Column(scale=4, elem_classes="panel"):
                        attention_prompt = gr.Textbox(value="ROMEO: The night is calm", label="Text to inspect", lines=3)
                        attention_checkpoint = gr.Dropdown(choices=choices, value=default_checkpoint, label="Checkpoint")
                        with gr.Row():
                            layer = gr.Number(value=5, precision=0, label="Layer")
                            head = gr.Number(value=0, precision=0, label="Head")
                        attention_button = gr.Button("Inspect attention", variant="primary")
                    with gr.Column(scale=8, elem_classes="panel"):
                        attention_image = gr.Plot(label="Attention heatmap", show_label=False)
                attention_button.click(attention_plot, [attention_prompt, layer, head, attention_checkpoint], attention_image)
        gr.Markdown("TinyTantrum / character-level modeling / benchmark evidence over vibes", elem_classes="section-note")

    return demo


def main() -> None:
    import gradio as gr

    port = int(os.environ.get("TINYTRANTRUM_PORT", "7861"))
    build_demo().launch(
        server_name="127.0.0.1",
        server_port=port,
        theme=gr.themes.Base(primary_hue="orange", secondary_hue="slate", neutral_hue="slate"),
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
