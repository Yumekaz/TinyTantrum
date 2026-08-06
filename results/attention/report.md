# TinyTantrum attention interpretability report

Checkpoint: `artifacts/checkpoints/full_run_best.pt`
Prompt: `ROMEO: / But soft, what light through yonder window breaks? / `
Device used for extraction: `cpu`

## What was measured

Every layer/head was scored by its average self-attention mass, previous-token mass, attention inside the previous 4 positions, long-range mass at distances of at least 8, and forbidden future-token mass.

The maximum future-token mass across all heads was `0.00000000`. This is the runtime evidence that the extracted maps respect the causal mask.

## Selected heads

| Selection rule | Layer | Head | Previous-token mass | Local-window mass | Long-range mass | Self mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Highest previous token | 0 | 2 | 0.9636 | 0.9474 | 0.0175 | 0.0177 |
| Highest local window | 0 | 0 | 0.0071 | 0.9495 | 0.0172 | 0.0322 |
| Highest long range | 5 | 4 | 0.0497 | 0.1543 | 0.6587 | 0.0938 |

## Interpretation limits

The selected heads are chosen by transparent statistics rather than visual cherry-picking. High previous-token or local-window mass is consistent with short-range routing; high long-range mass shows that a head uses distant context. These measurements do not prove that a head has a human-interpretable linguistic function. The report therefore treats them as behavioral evidence, not semantic claims.

Heatmaps:

- [Previous Token heatmap](previous_token_layer0_head2.png)
- [Local Window heatmap](local_window_layer0_head0.png)
- [Long Range heatmap](long_range_layer5_head4.png)
