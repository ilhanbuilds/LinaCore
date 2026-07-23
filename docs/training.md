# Training

LinaCore, Transformers/Torch tabanlı full SFT ve PEFT LoRA akışı sunar. Ağır
bağımlılıklar `.[training]`, LoRA için PEFT bağımlılığı
`.[training-lora]` extrasındadır; temel pakete eklenmez.

Config strict TOML’dur. Remote code ve telemetry kapalı, model local-only, overwrite ve
download varsayılan olarak kapalıdır. `smoke` bir step, `quick` üç epoch, `quality` ise daha
uzun hazırlık profilidir. Gerçek kullanılan config fingerprint ile metadata’ya yazılır;
device, dtype veya batch size sessizce değiştirilmez.

Pipeline config ve dataset doğrulama, deterministic split, tokenizer/model load,
assistant-only selective causal loss, periyodik validation, best-state seçimi, linear
warmup/decay, sınırlı checkpoint, atomic artifact save, manifest, model card
ve metadata aşamalarını çalıştırır. Başarısız run `.incomplete-*` dizininde `FAILED` olarak
kalır; başarılı artifact atomik olarak final adına taşınır.

LoRA target module adları modelin gerçek linear katmanlarından keşfedilir; trainable ve toplam
parametre sayısı metadata'ya yazılır. Native chat template varsa training aynı template'i
kullanır. Model context length ile training sequence length ayrı config alanlarıdır.

Ara optimizer checkpoint/resume henüz uygulanmamıştır. Final checkpoint metadata bunu
`resumable=false` olarak dürüstçe bildirir.
