# TinyTantrum

TinyTantrum is a reproducible, from-scratch character-level GPT laboratory. It implements the tokenizer, transformer blocks, causal attention, training loop, checkpoint recovery, benchmark evaluation, ablations, and generation path directly around PyTorch tensors and autograd.

This is a flagship ML engineering and experimental research artifact: the goal is not to claim a new transformer architecture, but to make every important training decision inspectable and every result falsifiable.

## Verified result

The reference configuration reached a validation loss of **1.4695779** using an independent 200-batch evaluation, against the reference value of approximately **1.4697**.

Two independent seeds were evaluated:

| Seed | Validation loss |
| ---: | ---: |
| 1337 | 1.4695779 |
| 2024 | 1.4792602 |

The context-length ablation also showed a clear trend under a fixed 2,000-step budget:

| Context length | Best validation loss |
| ---: | ---: |
| 64 | 1.5546164 |
| 128 | 1.4857790 |
| 256 | 1.4755781 |

See the detailed evidence in [results/benchmark.md](results/benchmark.md).

## What is implemented

- Deterministic character tokenizer and corpus split
- Hand-written embeddings, layer normalization, causal multi-head attention, feed-forward blocks, and residual connections
- AdamW training with warmup, cosine decay, gradient clipping, and live progress logging
- Atomic checkpoints containing model, optimizer, RNG, configuration, and step state
- Exact interrupted-vs-uninterrupted resume verification
- Independent evaluation and autoregressive generation commands
- Context-length ablation tooling
- Optional attention inspection and interactive dashboard

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts\train_model.py --steps 20 --layers 1 --heads 2 --embedding-size 64 --context-length 64 --batch-size 8 --dropout 0
```

The full reference run is intended for a GPU:

```bash
python scripts/train_model.py --steps 5000 --batch-size 64 --context-length 256 --layers 6 --heads 6 --embedding-size 384 --dropout 0.2 --seed 1337 --checkpoint checkpoints/full_run.pt --metrics runs/full_run_metrics.json
```

Evaluate a best checkpoint with the reference-style 200-batch estimate:

```bash
python scripts/evaluate_model.py --checkpoint checkpoints/full_run_best.pt --batches 200
```

Generate text:

```bash
python scripts/generate_text.py --checkpoint checkpoints/full_run_best.pt --prompt "ROMEO:" --tokens 400 --temperature 0.8 --top-k 20
```

## Dashboard

Install the optional UI dependencies:

```bash
python -m pip install -e ".[dashboard]"
python scripts\dashboard.py
```

The dashboard is a presentation layer over the same checkpoint and generation code; it is not part of the benchmark proof.

## Limitations

This is a small character-level model trained on Tiny Shakespeare. Generated text is qualitative evidence only. The context ablation controls training steps rather than equal compute, and the project does not claim a novel architecture or state-of-the-art language modeling result.
