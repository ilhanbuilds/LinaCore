# Backend’ler

`InferenceBackend` availability, format/device capability, load/unload, tokenizer creation, generation, streaming, cancellation, health, resource ve metadata sözleşmelerini tanımlar. Availability yalnız boolean değildir; package/version/missing dependency, gerekçe, device ve warning bilgisi taşır.

Transformers adapter optional ve lazy import edilir. Temel paket kurulumu Transformers/Torch indirmez. Dependency yoksa raw `ImportError` yerine Türkçe kurulum yönlendirmesi verilir. Gelecekte llama.cpp/GGUF, ONNX Runtime, DirectML ve özel Lina backend’leri aynı sınırı uygulamalıdır.

