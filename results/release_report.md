# TinyTantrum reproducibility and release record

Source commit: `e6c52f4e71a574a489dd3e4e3044e26f8e96b8f7`
Record version: `1`

## Dataset

- Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
- Local bytes: 1115394
- SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`
- Vocabulary: 65 characters
- Split: 90% train / 10% validation

## Model and training configuration

- Parameters: 10,770,816
- Layers / heads / embedding: 6 / 6 / 384
- Context length: 256
- Batch size: 64
- Dropout: 0.2
- Learning rate: 0.001
- Warmup / decay steps: 100 / 5000
- Optimizer: AdamW, beta2 `0.99`, weight decay `0.1`
- Precision: torch default dtype; no autocast

## Results

- Reference best checkpoint estimate: `1.4730365` at step `1750`
- Independent 200-batch evaluation: `1.4695779`
- Independent seed best estimate: `1.4847314` at step `1500`
- Benchmark target: `1.4697`

## Architectural ablation

Question: does learned positional information improve validation performance?

- Shared setup: 2000 steps, batch size 64, context 256, seed 1337
- With positions: `1.4626713` over 200 evaluation batches
- Without positions: `1.5620605` over 200 evaluation batches
- Difference: `+0.0993892` validation loss without positions

The result supports the conclusion that learned positional information materially helps this character-level model. The comparison is specific to this seed, dataset, architecture, and 2,000-step budget.


## Record-generation environment

- Python: `3.12.1`
- PyTorch: `2.13.0+cpu`
- Platform: `Windows-11-10.0.26200-SP0`
- Device: `CPU`
- CUDA: `not available`
- Default dtype: `torch.float32`

This snapshot describes the machine that generated this record. The historical training environment was not serialized with the benchmark metrics.

## Provenance limits

The machine-readable source of truth is `results/reproducibility.json`. Historical wall-clock duration was not preserved in the training metrics, and the raw checkpoint remains a separately supplied local artifact rather than a repository file.
