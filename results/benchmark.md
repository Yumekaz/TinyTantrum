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
