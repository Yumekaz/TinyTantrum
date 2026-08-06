# TinyTantrum benchmark result

## Reference run

Configuration: 6 layers, 6 heads, 384 embedding dimensions, context length 256, batch size 64, dropout 0.2, learning rate 1e-3, 5,000 steps, seed 1337.

## Result

The best checkpoint was selected during training and evaluated independently with 200 validation batches:

| Metric | Value |
| --- | ---: |
| Best checkpoint step | 1,750 |
| Validation loss | 1.4695779329538345 |
| Reference validation loss | 1.4697 |
| Absolute difference | approximately 0.00012 |

Status: **benchmark pass**.

The model also generated structured Shakespeare-like text from the prompt `ROMEO:`. Generated text is reported as a qualitative demonstration, not as the primary metric.

Hardware, runtime, PyTorch version, and the raw checkpoint should be attached to the final release record when publishing the run.

## Context-length ablation

Three matched runs used the same model, optimizer, seed, and 2,000 training steps while changing only the context length:

| Context length | Best validation loss | Best step |
| ---: | ---: | ---: |
| 64 | 1.5546164 | 2,000 |
| 128 | 1.4857790 | 2,000 |
| 256 | 1.4755781 | 1,750 |

The 256-character context performed best, improving validation loss by approximately 5.1% relative to context 64 and 0.7% relative to context 128. This supports the hypothesis that longer context helps this character-level model capture useful dependencies.

This comparison controls training steps, not total compute or tokens processed. The result therefore shows an empirical relationship under a fixed-step budget, not a claim that context 256 is universally compute-optimal.

## Multi-seed reproducibility

The reference configuration was trained independently with two seeds and evaluated using 200 validation batches:

| Seed | Validation loss |
| ---: | ---: |
| 1337 | 1.4695779 |
| 2024 | 1.4792602 |
| Mean | 1.4744191 |

The absolute difference between runs was approximately 0.00968. Both runs remained close to the reference, providing evidence that the benchmark result is not dependent on one unusually favorable initialization.
