# Evaluation

Training raporu final/validation loss ve güvenli hesaplanabilirse perplexity içerir.
`data/examples/lina_eval_v0.1.jsonl` dokuz davranış promptu tanımlar. Comparison runner base
ve fine-tuned çıktıyı, token kullanımını, latency’yi, keyword kuralını ve boş `human_review`
alanını aynı typed JSON raporda saklar.

```console
python scripts/compare_local_models.py models/manual-smollm2-135m/linacore-model.toml models/lina-135m-sft-v0.1/linacore-model.toml
```

Keyword skoru insan değerlendirmesi değildir. İlk 3-epoch koşuda generation success %100
olsa da fine-tuned rule pass `1/9`, base `2/9` olmuştur. Model teknik olarak çalışır, fakat
küçük dataset henüz güvenilir Lina davranışı üretmemektedir.

v0.2 holdout 90 prompt ve 30 kategori içerir. Değerlendirme required concept gruplarını,
forbidden claim'i, tekrarı, soru kopyalamayı, boş çıktıyı, Türkçe sezgisini ve
kritik failure mantığını birlikte uygular. Base, v0.1 ve v0.2 aynı temperature 0,
max_new_tokens 64 ve seed 42 ile karşılaştırılır. Sonuçlar ve sınırlamalar
[Lina Model v0.2 belgesinde](lina-model-v0.2.md) bulunur.
