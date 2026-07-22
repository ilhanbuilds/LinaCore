# Configuration

`AppConfig` immutable dataclass ağacıdır ve UTF-8 TOML’dan yüklenir. Malformed TOML, unknown field, aşırı büyük dosya ve geçersiz değerler `ConfigurationError` üretir. `runtime`, `backend.transformers`, `generation` ve `security` bölümleri v0.2 politikalarını taşır.

Transformers varsayılanları local-only, remote code kapalı ve remote download kapalıdır. Model root, concurrency ve timeout değerleri typed doğrulanır. Environment override alanı bilinçli olarak `LINA_CORE_LOG_LEVEL`, `LINA_CORE_PROVIDER` ve `LINA_CORE_TOKENIZER` ile sınırlıdır; `.env` LinaCore tarafından okunmaz. Secret değerler config/log içine konulmamalıdır.

