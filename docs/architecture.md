# Mimari

Bağımlılıklar dış katmanlardan iç sözleşmelere doğru ilerler. CLI, diagnostics ve gelecekteki transport adapter’ları application service’lerini çağırır; runtime model/tokenizer provider sözleşmelerine dayanır. Backend adapter’ları optional framework ayrıntılarını izole eder. `core` ve `security`, Lina Desktop veya belirli bir backend import etmez.

Runtime lifecycle, provider registry, aktif model/tokenizer, cancellation ve resource sayaçlarını yönetir. Inference request/response/stream event modelleri transport bağımsızdır. Prompt ve context budgeting provider içine gömülmez. Model discovery yalnızca manifest okur; ağırlık yükleme lazy olarak açık runtime komutunda gerçekleşir.

Transformers ilk gerçek adapter’dır; sözleşme gelecekte llama.cpp, GGUF, ONNX Runtime, DirectML, CUDA/CPU ve özel Lina runtime implementasyonlarına açıktır. Serving katmanı raw exception döndürmez.

