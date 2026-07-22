# İlk Lina modeli

`Lina-135M-SFT-v0.1`, `HuggingFaceTB/SmolLM2-135M` revision
`93efa2f097d58c2a74874c7e644dbc9b0cee75a2` üzerine Lina SFT v0.1 ile uygulanan full
supervised fine-tuning sonucudur. Sıfırdan pretrained değildir.

Doğrulanan 2026-07-22 quick koşusu CPU/float32 üzerinde 3 epoch ve 60 step sürdü. Süre
60.15 saniye, final training loss `1.1212`, validation loss `3.6084`, perplexity `36.91`
oldu. Validation sonucunun training loss’a göre kötüleşmesi overfitting işaretidir.

Artifact ağırlıkları public repository’ye eklenmez. Local dizinde model/tokenizer,
`linacore-model.toml`, Türkçe `MODEL_CARD.md`, training/checkpoint metadata ve raporlar
üretilir. Compatibility kontrolleri 8/8 geçmiştir. Davranış karşılaştırması kalite artışı
kanıtlamamıştır; bu model pipeline MVP’si ve araştırma artifact’idir.

Base model Apache-2.0 lisanslıdır. Türetilmiş ağırlık paylaşımında lisans/notice metni
korunmalı, değişiklik belirtilmeli ve üçüncü taraf yükümlülükleri ayrıca değerlendirilmelidir.
