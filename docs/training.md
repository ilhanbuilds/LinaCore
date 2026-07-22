# Training

Version 0.1 performs dry-run validation only. It parses a typed configuration, validates positive hyperparameters and precision, resolves manifest/output paths inside the workspace, validates the manifest and reports Python/platform details. It does not allocate a GPU, load weights or train.

A future stack must be selected through measured needs. Reproducibility requires recorded seeds, immutable manifests, environment capture, checksummed inputs and metrics. Checkpoint policies must bound retention, verify integrity, document license and distinguish resumable state from exported weights.

