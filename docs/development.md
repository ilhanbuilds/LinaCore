# Geliştirme

Python 3.11+ virtual environment oluşturun. Temel paket ağır ML dependency içermez; development için `.[dev]`, adapter geliştirme için `.[inference-transformers,inference-torch]` veya `.[inference-llama-cpp]` kurun.

Her değişiklikte `pytest -m "not manual_integration and not model_integration"`,
`ruff check .`, `mypy src/lina_core`, `compileall`, `pip check`, `build` ve
`python scripts/audit_public_repository.py` çalıştırın. CI bu kapıları Python 3.11 ve
3.12 üzerinde, yalnız `.[dev]` kurulu hafif ortamda ve matrix fail-fast kapalı olarak
çalıştırır. Provider conformance suite yeni backend’lerde tekrar kullanılmalıdır.
Gerçek model testleri `model_integration` marker’ıyla işaretlenir ve varsayılan ortamda
skip edilir.

Mypy, optional runtime paketleri kurulu değilken de LinaCore kaynaklarını strict modda
denetler. Yalnız `numpy`, `torch` ve `transformers` import çözümlemesi için dar kapsamlı
override vardır; proje içi modüller ve diğer üçüncü taraf paketler global olarak
yoksayılmaz.

Gerçek model indirme, training, GGUF export ve ağır artifact gerektiren kontroller
`manual_integration` kapsamındadır; normal CI bunları çalıştırmaz.

Manual Transformers doğrulaması için önce [model compatibility](model-compatibility.md) akışını izleyin. İndirme açık `--allow-download` onayı ister; CI ve varsayılan testler bu script’i çalıştırmaz.

Identifier, kaynak dosya ve commit mesajları İngilizce; README/docs ve CLI kullanıcı mesajları Türkçedir. Conventional Commits kullanın. Secret, kullanıcı verisi, model ağırlığı, checkpoint, generated report ve log commit etmeyin. Yeni dependency somut gerekçe gerektirir.
