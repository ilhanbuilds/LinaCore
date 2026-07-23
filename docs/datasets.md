# Dataset

`data/examples/lina_sft_v0.1.jsonl`, 24 güvenli ve sentetik konuşma kaydı içerir. Her kayıt
strict `id`, `messages`, `category`, `language`, `source`, `license`, `quality_status` ve
`metadata` alanlarını taşır. Örnek veri CC0-1.0 olarak sunulur; kişisel veya kullanıcıya özel
veri içermez.

Validator boş/geçersiz mesajı, unknown field’ı, duplicate id/içeriği, aşırı boyutu,
secret/token benzeri metni, kişisel yolu, e-posta ve telefon benzeri veriyi reddeder.
Fingerprint canonical UTF-8 JSON üzerinden SHA-256’dır. Split seed ile deterministiktir ve
küçük dataset için validation/test fallback uygular.

`data/examples/lina_sft_v0.2.jsonl` 30 kategoride 600 sentetik kayıt içerir. Her kategori
20 kayıtla dengelidir; 540 kayıt Türkçe, 60 kayıt İngilizcedir. Grouped split aynı
`template_group` değerini tek split'te tutar ve 480/60/60 dağılımı üretir. Ayrı
`lina_behavior_holdout_v0.2.jsonl` 90 prompt içerir. Kalite audit'i normalize/near duplicate,
cevap şablonu tekrarı, kategori dengesi ve cross-split/holdout leakage kontrolü yapar.

```console
python -m lina_core dataset validate data/examples/lina_sft_v0.1.jsonl
python -m lina_core dataset stats data/examples/lina_sft_v0.1.jsonl
python -m lina_core dataset fingerprint data/examples/lina_sft_v0.1.jsonl
python -m lina_core dataset split data/examples/lina_sft_v0.1.jsonl --seed 42
python -m lina_core dataset render data/examples/lina_sft_v0.1.jsonl --limit 3
```

Dataset v0.2 deterministic olarak yeniden üretilebilir:

```console
python scripts/generate_lina_dataset_v0_2.py
```
