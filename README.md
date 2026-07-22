# LinaCore

LinaCore, Lina masaüstü uygulaması için geliştirilen bağımsız ve yerel yapay zekâ çekirdeğidir. `0.2.0a0` sürümü, güvenli local model discovery ve optional Transformers adapter’ı üzerinden ilk gerçek yerel inference yolunu sağlar. Projede model ağırlığı veya veri seti bulunmaz.

## Amaç ve mevcut durum

Lina Desktop arayüz, cihaz entegrasyonu, geçmiş ve kullanıcı ayarlarından; LinaCore ise model/tokenizer sözleşmeleri, inference, streaming, model yükleme, değerlendirme ve güvenlik sınırlarından sorumludur. Projeler gelecekte local protocol ile bağlanacak, fakat birbirlerine veya Ollama’ya zorunlu olarak bağımlı olmayacaktır.

v0.2.0-alpha kapsamında mock provider korunmuş; backend abstraction, local-only Transformers adapter, model manifest/discovery, device ve dtype seçimi, prompt/context yönetimi, cancellation, diagnostics ve benchmark sözleşmesi eklenmiştir. Gerçek model çalıştırmak için kullanıcı kendi modelini ve lisans uygunluğunu sağlamalıdır.

## Gereksinimler ve kurulum

Python 3.11 veya üzeri gerekir. Temel kurulum GPU, Torch, Transformers, Docker ya da Ollama indirmez:

```console
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Transformers tabanlı gerçek inference için optional grupları ayrıca kurun:

```console
python -m pip install -e ".[inference-transformers,inference-torch]"
```

Geliştirme araçları:

```console
python -m pip install -e ".[dev]"
```

## Model ağırlıkları ve manifest

Model ağırlıkları Git’e eklenmez. Her model, yapılandırılmış `models/` root’u altında bir `linacore-model.toml` manifestiyle tanımlanır. Manifest model kimliği, backend, göreli model yolu, lisans, context sınırı, dtype/device politikası, capability ve isteğe bağlı SHA-256 checksum bilgilerini taşır.

[Örnek manifest](examples/models/example-transformers/linacore-model.toml) yalnızca biçimi gösterir; gerçek veya kullanılabilir bir model değildir. Her modelin lisansını, kaynağını ve kullanım koşullarını ayrıca inceleyin.

## Discovery ve güvenlik varsayılanları

Discovery yalnızca yapılandırılmış model root’u içinde manifest arar; ağırlık yüklemez, inference başlatmaz ve ağ erişimi yapmaz. Varsayılanlar:

- `local_files_only=true`
- `trust_remote_code=false`
- remote download kapalı
- path traversal ve symlink escape reddedilir
- unknown manifest/config alanları reddedilir
- unsupported capability, dtype ve quantization kontrollü hata üretir

## Device ve dtype

`auto`, `cpu`, `cuda` ve `cuda:N` politikaları desteklenir; MPS yalnızca platform sağlıyorsa raporlanır. CPU güvenli fallback’tir. CPU üzerinde `float16` sessizce uygulanmaz. `int8` ve `int4` typed foundation olarak tanımlıdır, ancak bu sürümde quantization dependency’si zorunlu değildir ve destek yoksa açık hata döner.

## CLI

```console
lina-core version
lina-core health
lina-core diagnostics
lina-core diagnostics --json
lina-core backends list
lina-core devices list
lina-core models discover
lina-core models list
lina-core models validate <model-id-or-manifest>
lina-core runtime smoke
lina-core runtime load <model-id>
lina-core inference generate --model <model-id> --prompt "Merhaba" --max-new-tokens 32
lina-core inference stream --model <model-id> --prompt "Merhaba"
```

CLI argümanı olarak verilen prompt shell history içinde kalabilir. Hassas veri için prompt argümanı kullanmayın. Kalıcı runtime/network service bu sürümün kapsamında değildir; `runtime load` yalnızca komut süresince yükleme doğrulaması yapar.

## Generation, streaming ve cancellation

Generation seçenekleri backend’e kontrollü biçimde eşlenir. `temperature=0` greedy çalışır; seed Python, NumPy mevcutsa NumPy ve Torch’a uygulanır. Token usage tokenizer sonucundan, latency monotonic saatten hesaplanır. Streaming sırası `started → token* → completed/cancelled/error` şeklindedir. Request bazlı cancellation ve shutdown sırasında toplu iptal runtime API’sinde bulunur.

## Diagnostics ve benchmark

Diagnostics; LinaCore/Python sürümü, işletim sistemi, mimari, optional package durumu, device’lar, model sayısı, yazma izni ve güvenlik varsayılanlarını raporlar. Secret veya environment variable değeri göstermez.

Benchmark sözleşmesi load time, first-token latency, total latency, token sayıları, tokens/s ve optional memory alanlarını tanımlar. Mock sonuçları gerçek benchmark sayılmaz; CLI benchmark komutu yüklü gerçek model yoksa kontrollü hata verir.

## Test ve geliştirme

```console
python -m pytest
python -m pytest -m "not model_integration"
python -m ruff check .
python -m mypy src/lina_core
python -m compileall -q src
python -m build
```

Gerçek model entegrasyon testi `model_integration` marker’ı altındadır ve kullanıcı model yolu/optional dependency sağlamadıkça skip edilir.

## Mimari belgeler

- [Mimari](docs/architecture.md)
- [Backend’ler](docs/backends.md)
- [Modeller](docs/models.md)
- [Inference](docs/inference.md)
- [Donanım](docs/hardware.md)
- [Runtime](docs/runtime.md)
- [Configuration](docs/configuration.md)
- [Security](docs/security.md)
- [Dataset](docs/datasets.md)
- [Training](docs/training.md)
- [Evaluation](docs/evaluation.md)
- [Model yol haritası](docs/model-roadmap.md)
- [Geliştirme](docs/development.md)

## Bilinen sınırlamalar ve yol haritası

Bu sprintte repository içinde gerçek model, production server, Lina Desktop entegrasyonu, fine-tuning, vision, embeddings, tool calling ve hazır quantization yoktur. Transformers adapter gerçek local model için hazırlanmıştır; fakat bu release doğrulamasında model ağırlığı çalıştırılmamıştır. Sonraki aşamalar GGUF/llama.cpp adapter’ı, küçük local model uyumluluk matrisi, Türkçe değerlendirme ve reproducible fine-tuning araştırmasıdır.

## Lisans

Kod MIT lisanslıdır. Model, tokenizer ve veri setlerinin lisansları bağımsızdır; bir model ağırlığı LinaCore ile kullanıldığı için otomatik olarak MIT kapsamına girmez.

