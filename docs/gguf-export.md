# GGUF export

Export contract local Transformers artifact’ini llama.cpp `convert_hf_to_gguf.py` aracıyla
F16 veya F32 GGUF’a dönüştürür. Araç otomatik indirilmez; yolu açıkça verilmelidir.
Subprocess `shell=false` ve sabit argüman listesi kullanır. Kaynak/output model root dışına
çıkamaz, mevcut output açık overwrite olmadan değiştirilemez.

Bu ortamda conversion aracı, GGUF artifact’i ve `llama-cpp-python` bulunmadığı için gerçek
export/inference doğrulanmamıştır. Q8_0 ve Q4_K_M quantization sonraki ayrı aşamadır.
