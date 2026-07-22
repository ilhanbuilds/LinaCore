# Geliştirme

Python 3.11+ virtual environment oluşturun. Temel paket ağır ML dependency içermez; development için `.[dev]`, adapter geliştirme için `.[inference-transformers,inference-torch]` kurun.

Her değişiklikte `pytest`, `pytest -m "not model_integration"`, `ruff check .`, `mypy src/lina_core`, `compileall`, `pip check` ve `build` çalıştırın. Provider conformance suite yeni backend’lerde tekrar kullanılmalıdır. Gerçek model testleri `model_integration` marker’ıyla işaretlenir ve varsayılan ortamda skip edilir.

Identifier, kaynak dosya ve commit mesajları İngilizce; README/docs ve CLI kullanıcı mesajları Türkçedir. Conventional Commits kullanın. Secret, kullanıcı verisi, model ağırlığı, checkpoint, generated report ve log commit etmeyin. Yeni dependency somut gerekçe gerektirir.
