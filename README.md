# LinaCore

LinaCore is the local, provider-independent AI core planned for the Lina desktop application. This `0.1.0a0` release establishes a small, typed and testable Python foundation; it does **not** contain or claim a real language model.

## Purpose and relationship to Lina

Lina Desktop owns the user experience, device integration, history and preferences. LinaCore owns model/tokenizer contracts, inference, streaming, loading, future training and evaluation, safety boundaries and a local protocol. The projects are deliberately decoupled: LinaCore has its own CLI, configuration, tests and documentation. A future local protocol will connect them without making Ollama a required dependency.

## Current status and features

The foundation includes immutable TOML configuration, environment overrides, lifecycle and logging, provider registries, deterministic mock inference and streaming, dataset validation, training dry-run, lightweight evaluation, bounded reports, security controls, an in-process serving adapter and a JSON CLI. The mock provider exists only to test architecture and makes no AI-quality claim.

## Architecture and project structure

`src/lina_core` is split into `core`, `runtime`, `inference`, `models`, `tokenizer`, `datasets`, `training`, `evaluation`, `serving` and `security`. Business logic does not depend on a transport or Lina Desktop. See [architecture](docs/architecture.md).

## Requirements and installation

Python 3.11 or newer is required. No GPU, Docker, Ollama or runtime dependency is required.

```console
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## Development environment and testing

```console
python -m pytest
python -m ruff check .
python -m mypy src/lina_core
python -m build
```

## CLI usage

```console
lina-core version
lina-core health
lina-core config validate configs/development.toml
lina-core models list
lina-core runtime smoke
lina-core dataset validate path/to/manifest.json
lina-core training dry-run configs/training.example.toml
lina-core evaluation run configs/evaluation.example.toml
```

`runtime start` explicitly reports that a persistent network service is not implemented. `runtime smoke` loads the mock provider, performs deterministic inference and shuts down cleanly.

## Configuration

Defaults are safe for local development. TOML fields are typed and unknown fields fail closed. Only `LINA_CORE_LOG_LEVEL`, `LINA_CORE_PROVIDER` and `LINA_CORE_TOKENIZER` are recognized as basic environment overrides. Secrets do not belong in project configuration. See [configuration](docs/configuration.md).

## Models and tokenizer

There is no real model yet. The built-in mock model and reversible UTF-8 byte tokenizer validate contracts only. Model weights are never stored in Git. Future model licenses and weights are not automatically covered by this repository's license.

## Datasets

No training dataset is included or downloaded. Dataset content is ignored by Git and requires provenance, license, privacy review, cleanup, validation and a checksummed manifest. See [datasets](docs/datasets.md).

## Training

Training is dry-run only in this release: it validates configuration, manifest and bounded output paths and emits an environment report. It never starts a training loop or creates a checkpoint. See [training](docs/training.md).

## Evaluation

Evaluation supports plumbing-level deterministic metrics and bounded JSON reports against the mock provider. Results are not quality benchmarks. See [evaluation](docs/evaluation.md).

## Security and privacy

Workspace containment, traversal/symlink escape checks, resource limits, conservative secret detection and log redaction form the initial boundary. This is defense in depth, not a complete sandbox. See [security](docs/security.md).

## Roadmap and known limitations

The next stages are an openly licensed baseline adapter, tokenizer research, licensed Turkish data preparation, reproducible fine-tuning, evaluation, quantization and ultimately a standalone Lina model. Today there is no real model, HTTP/IPC server, training loop, model download or production security guarantee. See [model roadmap](docs/model-roadmap.md).

## Contributing

Use focused changes, type annotations and tests. Run format/lint/type/test/build checks before committing. Never commit secrets, user data, weights, checkpoints, generated reports or logs. See [development](docs/development.md).

## License

Code is licensed under the MIT License. Any future models and datasets require separate license review; model weights do not automatically fall under MIT.

