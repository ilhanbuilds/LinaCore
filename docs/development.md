# Geliştirme

Python 3.11+ virtual environment oluşturun. Temel paket ağır ML dependency içermez; development için `.[dev]`, adapter geliştirme için `.[inference-transformers,inference-torch]` veya `.[inference-llama-cpp]` kurun.

Her değişiklikte `pytest`, `pytest -m "not model_integration"`, `ruff check .`, `mypy src/lina_core`, `compileall`, `pip check`, `build` ve `python scripts/audit_public_repository.py` çalıştırın. Provider conformance suite yeni backend’lerde tekrar kullanılmalıdır. Gerçek model testleri `model_integration` marker’ıyla işaretlenir ve varsayılan ortamda skip edilir.

Gerçek model indirme, training, GGUF export ve ağır artifact gerektiren kontroller
`manual_integration` kapsamındadır; normal CI bunları çalıştırmaz.

Manual Transformers doğrulaması için önce [model compatibility](model-compatibility.md) akışını izleyin. İndirme açık `--allow-download` onayı ister; CI ve varsayılan testler bu script’i çalıştırmaz.

Identifier, kaynak dosya ve commit mesajları İngilizce; README/docs ve CLI kullanıcı mesajları Türkçedir. Conventional Commits kullanın. Secret, kullanıcı verisi, model ağırlığı, checkpoint, generated report ve log commit etmeyin. Yeni dependency somut gerekçe gerektirir.
