# Lina Model v0.2

`Lina-0.5B-SFT-v0.2`, Apache-2.0 lisanslı `Qwen/Qwen2.5-0.5B-Instruct`
revision `7ae557604adf67be50417f59c2c2f167def9a775` üzerine LoRA SFT uygulanmış
deneysel bir modeldir. Sıfırdan pretrained değildir ve ağırlıklar repository'de
yayımlanmaz.

## Sprint sonucu: PARTIAL

Bağımsız 90-prompt holdout'ta aynı deterministic generation ayarlarıyla:

| Model | Skor | Kritik failure | Generation |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct base | 4/90 (%4,44) | 21 | %100 |
| Lina-135M-SFT-v0.1 | 2/90 (%2,22) | 21 | %100 |
| Lina-0.5B-SFT-v0.2 | 45/90 (%50,00) | 12 | %100 |

v0.2 iki baseline'dan belirgin biçimde daha iyi olsa da minimum %60 kabul eşiğini ve
sıfır kritik failure koşulunu karşılamadı. Bu nedenle production-ready değildir,
LinaCore package sürümü `0.3.0a0` olarak kaldı ve başarı iddiası yapılmadı.

## Dataset ve split

- 600 tamamen sentetik, CC0-1.0 kayıt
- 30 kategori, kategori başına 20 kayıt
- 540 Türkçe ve 60 İngilizce kayıt
- Grouped ve kategori-korumalı split: 480 / 60 / 60
- Bağımsız holdout: 90 prompt
- Fingerprint: `sha256:20adc6303f85e171447ebc339c9a879b1a170293cc425fd47fa1c51898b41205`
- Exact/normalized/near duplicate, template overuse ve split/holdout leakage: 0
- Token uzunluğu 73–171; `max_sequence_length=256` ile truncation: 0

## Eğitim

- CPU / float32, 28 dakika 26,8 saniye
- LoRA r=8, alpha=16, dropout=0,05
- Target: `q/k/v/o_proj`, `gate/up/down_proj`; model linear katmanlarından keşfedildi
- Trainable: 4.399.104 / 498.431.872 parametre (%0,883)
- 720 micro-step, 180 optimizer step, best step 720
- Final train loss: 0,03711
- Validation loss: 0,39995
- Perplexity: 1,4917

Validation loss 2,3516'dan 0,3999'a düzenli indi. Buna rağmen serbest generation ve
kritik davranış sonuçları yetersiz kaldı; loss tek başına kalite kanıtı değildir.
Kritik kategorileri iki kat ağırlıklandıran ayrı 600-step aday da denendi; 22/90 ve
14 kritik failure ile daha kötü olduğu için seçilmedi ve artifact'i silindi.

## v0.1 kök neden analizi

v0.1'de 24 kaydın yalnız 20'si train split'e girdi; kategori başına çoğunlukla tek
örnek vardı. Train loss 1,1212 iken validation loss 3,6084 ve perplexity 36,91 oldu.
Base 2/9 iken tuned model 1/9 aldı. 24 kaydın token uzunluğu 60–199 arasındaydı ve
`max_length=128` bir kaydı truncate etti. 135M kapasite, düşük çeşitlilik, küçük
validation ve tekrar eden yanıt biçimi birlikte overfitting oluşturdu. Eski manifest ayrıca
modelin 8.192 context'i yerine training sequence length 128'i yazıyordu; yeni runner bu iki
kavramı ayırır.

## Tekrarlama ve kullanım

Model indirme ve gerçek training manual/local akıştır; varsayılan CI bunları çalıştırmaz.

```powershell
python -m lina_core train plan `
  --config configs/training/lina-0.5b-sft-v0.2.toml `
  --profile quality
```

Yerel artifact mevcutsa:

```powershell
python -m lina_core inference generate `
  --model lina-0.5b-sft-v0.2 `
  --prompt "Merhaba, sen kimsin?" `
  --max-new-tokens 96 `
  --temperature 0 `
  --seed 42
```

GGUF export edilmedi. Bir sonraki kalite çalışması daha doğal ve durum-temelli veri,
insan review kalibrasyonu ve daha az şablon tekrarına odaklanmalıdır.
