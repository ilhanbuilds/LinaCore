# Backend’ler

`InferenceBackend` availability, format/device capability, load/unload, tokenizer creation, generation, streaming, cancellation, health, resource ve metadata sözleşmelerini tanımlar. Availability yalnız boolean değildir; package/version/missing dependency, gerekçe, device ve warning bilgisi taşır.

Transformers adapter optional ve lazy import edilir. Temel paket kurulumu Transformers/Torch indirmez. Dependency yoksa raw `ImportError` yerine Türkçe kurulum yönlendirmesi verilir.

`LlamaCppBackend`, aynı sözleşmeye uyan küçük ve mock destekli bir foundation’dır. Optional `llama-cpp-python` availability raporu, güvenli GGUF config’i, load/unload yaşam döngüsü ve enjekte edilebilir test engine’i içerir. Paket kurulu olsa bile production engine entegrasyonu açıkça devre dışıdır; gerçek GGUF inference tamamlanmış gibi sunulmaz.

GGUF export contract’ı açıkça sağlanan llama.cpp conversion aracını güvenli subprocess
argümanlarıyla çalıştırabilir. Bu sprint ortamında conversion aracı/GGUF bulunmadığından gerçek
llama.cpp inference sonucu yoktur.

ONNX Runtime, DirectML ve özel Lina backend’leri gelecekte aynı sınırı uygulamalıdır.
