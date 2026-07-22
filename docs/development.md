# Development

Create a Python 3.11+ virtual environment and install `.[dev]`. Run `pytest`, `ruff check .`, `mypy src/lina_core`, `compileall`, `pip check` and `build` before publishing. Keep modules focused, public boundaries typed and platform assumptions explicit.

Use conventional, focused commits (`feat:`, `fix:`, `test:`, `docs:`, `chore:`). Do not commit environment files, credentials, user data, models, checkpoints, reports or logs. New dependencies need a concrete justification and must not make GPU or Docker mandatory.
