# LinaCore

**Yerel, provider-bağımsız ve güvenli yapay zekâ çalışma zamanı temeli.**

LinaCore; model keşfi, yükleme, metin üretimi, streaming, iptal, sağlık ve kaynak
raporlama için typed sözleşmeler sunar. Temel kurulum ağır ML paketleri indirmez; gerçek
backend’ler isteğe bağlıdır. Projede model ağırlığı, cache veya veri seti bulunmaz.

> **Proje durumu:** Transformers local inference ve ilk Lina-specific SFT pipeline’ı
> çalışmaktadır. `Lina-135M-SFT-v0.1`, SmolLM2-135M üzerine küçük sentetik veriyle eğitilmiş
> deneysel bir türevdir; sıfırdan pretrained bir Lina foundation modeli değildir. GGUF export
> sözleşmesi hazır, gerçek export ve llama.cpp doğrulaması ise henüz tamamlanmamıştır.

## Öne çıkanlar

- Tek `InferenceBackend` sınırı üzerinden mock, Transformers ve llama.cpp foundation
- Güvenli, local-only model manifesti ve discovery; path traversal/symlink escape koruması
- Generation, streaming, cancellation, stop sequence, context budget ve typed hata modeli
- Optional dependency, cihaz, dtype, sağlık ve kaynak diagnostics
- İnternet/model gerektirmeyen varsayılan testler; gerçek model için ayrı manual integration
- Strict Türkçe SFT dataset’i, deterministik split/fingerprint ve atomik model artifact üretimi
- Base/fine-tuned davranış karşılaştırması, model card ve compatibility raporu
- Hafif çekirdek: temel `lina-core` paketinin runtime dependency’si yoktur

## Mimari

```text
CLI / gelecekteki local protocol
             │
     runtime ve typed modeller
             │
       InferenceBackend
       ├── MockProvider
       ├── TransformersBackend     (gerçek local inference)
       └── LlamaCppBackend         (mock destekli foundation)
             │
     manifest / discovery / security
```

Lina Desktop; arayüz, cihaz entegrasyonu, geçmiş ve kullanıcı ayarlarından sorumludur.
LinaCore model/tokenizer sözleşmelerini ve inference sınırını taşır. İki proje gelecekte
local protocol ile bağlanabilir; LinaCore Ollama’ya zorunlu olarak bağımlı değildir.

## Kurulum

Python 3.11 veya üzeri gerekir:

```console
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e .
```

Gerçek Transformers inference için:

```console
python -m pip install -e ".[inference-transformers,inference-torch]"
```

llama.cpp paketi yalnız foundation geliştirmesi için ayrı kurulabilir; bu sürüm gerçek
GGUF üretimini etkinleştirmez:

```console
python -m pip install -e ".[inference-llama-cpp]"
```

Geliştirme araçları için `python -m pip install -e ".[dev]"` kullanın.

## CLI örnekleri

```console
lina-core version
lina-core health
lina-core diagnostics --json
lina-core backends list
lina-core devices list
lina-core models discover
lina-core models validate <model-id-or-manifest>
lina-core runtime smoke
lina-core inference generate --model <model-id> --prompt "Merhaba" --max-new-tokens 32
lina-core inference stream --model <model-id> --prompt "Merhaba"
```

Komut satırındaki prompt shell history’de kalabilir; hassas veri vermeyin. Kalıcı
runtime/network service henüz kapsamda değildir. `runtime load` yalnız komut süresince
yükleme doğrulaması yapar.

## Model manifesti ve güvenlik

Her model, yapılandırılmış `models/` root’u altında bir `linacore-model.toml` ile tanımlanır.
Manifest; backend, göreli yollar, kaynak revision’ı, lisans, context, dtype/device,
capability ve isteğe bağlı SHA-256 bilgilerini taşır. Transformers ve GGUF için örnekler:

- [Transformers manifesti](examples/models/example-transformers/linacore-model.toml)
- [GGUF manifesti](examples/models/example-gguf/linacore-model.toml)

Örnekler kullanılabilir model içermez. Varsayılanlar `local_files_only=true`,
`trust_remote_code=false` ve remote download kapalıdır. Model indirme yalnız açık onay
gerektiren manual script ile yapılır. Her modelin lisansını ve kaynağını ayrıca doğrulayın.

## Gerçek model doğrulaması

Manual compatibility akışı; load, generation, streaming, cancellation, stop sequence,
health, resource reporting ve unload davranışlarını typed JSON rapora kaydeder. Varsayılan
test koşusu internete erişmez ve model ağırlığı istemez.

2026-07-22 tarihinde `HuggingFaceTB/SmolLM2-135M`, revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, Apache-2.0, CPU/float32 üzerinde sekiz
kontrolün tamamını geçti. Bu bir kalite/benchmark iddiası değil, backend uyumluluk
doğrulamasıdır. Tekrarlama adımları ve ölçümler için
[model compatibility belgesine](docs/model-compatibility.md) bakın.

## İlk deneysel Lina modeli

Dataset’i doğrulayıp training planını görmek ve cached base modelle açık onaylı quick eğitim
başlatmak için:

```console
python -m lina_core dataset validate data/examples/lina_sft_v0.1.jsonl
python -m lina_core train plan --config configs/training/lina-135m-sft-v0.1.toml --profile quick
python -m lina_core train run --config configs/training/lina-135m-sft-v0.1.toml --profile quick --yes
```

Bu repository model ağırlığını içermez. Yerel artifact `models/lina-135m-sft-v0.1` altında
üretilir ve Git tarafından ignore edilir. Ayrıntılar: [ilk Lina modeli](docs/first-lina-model.md).

## Test durumu

```console
python -m pytest -m "not model_integration"
python -m ruff check .
python -m mypy src/lina_core
python -m compileall -q src
python -m build
python scripts/audit_public_repository.py
```

`model_integration` marker’ı kullanıcı tarafından sağlanan local manifest ve optional ML
dependency ister; normal koşuda skip edilir. Güncel kesin test sayısı release yerine CI/test
çıktısından izlenir.

## Yol haritası

- Lina SFT dataset’ini büyütmek ve assistant davranış değerlendirmesini iyileştirmek
- Güvenli ara checkpoint/resume desteği eklemek
- GGUF export’u gerçek llama.cpp aracı ve engine ile doğrulamak
- Açık lisanslı küçük modellerle compatibility matrisini genişletmek
- Türkçe tokenizer/dataset lisans ve provenance araştırması
- Versioned Türkçe automatic/human evaluation
- Reproducible fine-tuning ve checkpoint politikası
- DirectML/ONNX ve farklı local donanım hedefleri

## Bilinen sınırlamalar

- llama.cpp/GGUF yalnız availability, config, discovery, manifest ve mock engine temelidir.
- Production server ve Lina Desktop entegrasyonu yoktur.
- Hazır quantization, fine-tuning, vision, embeddings ve tool calling yoktur.
- Model çıktısının doğruluğu veya güvenliği garanti edilmez.
- Pre-alpha public API’lerde stabilite garantisi yoktur.

## Dokümantasyon ve katkı

[Mimari](docs/architecture.md) · [Backend’ler](docs/backends.md) ·
[Modeller](docs/models.md) · [Inference](docs/inference.md) ·
[Security](docs/security.md) · [Geliştirme](docs/development.md) ·
[Model yol haritası](docs/model-roadmap.md)

Katkı süreci için [CONTRIBUTING.md](CONTRIBUTING.md), güvenlik bildirimi için
[SECURITY.md](SECURITY.md) dosyasını okuyun.

## Lisans

Kod [MIT](LICENSE) lisanslıdır. Model, tokenizer ve veri seti lisansları bağımsızdır;
LinaCore ile kullanılmaları onları MIT kapsamına sokmaz.
