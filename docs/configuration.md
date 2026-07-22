# Configuration

`AppConfig` is an immutable dataclass loaded from UTF-8 TOML. Defaults select mock providers and bounded resources. Unknown fields, malformed TOML, oversized files and invalid values fail with `ConfigurationError`. Development TOML mirrors the typed structure.

The minimal environment override surface is `LINA_CORE_LOG_LEVEL`, `LINA_CORE_PROVIDER` and `LINA_CORE_TOKENIZER`. Environment variables are read explicitly; `.env` is not parsed by LinaCore. Credentials and secrets must use a future dedicated secret store and must never be logged or committed.

