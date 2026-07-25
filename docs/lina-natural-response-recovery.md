# Lina Natural Response Recovery

Bu belge, Lina'nın kullanıcı sorusuna ilk cümlede doğal ve doğrudan cevap verme
yeteneğini kurtarmak için yürütülen curated dataset ve curriculum canary sprintinin
kanıtlarını kaydeder.

**Karar: FAILED.** Curriculum Stage 1 üç kontrollü denemede de zorunlu 7/8
kapısını geçemedi. Bu nedenle Stage 2 ve tam `Lina-0.5B-SFT-v0.2.1` eğitimi
başlatılmadı. LinaCore package sürümü `0.3.0a0` olarak kaldı.

## Önceki modelde görülen sorun

Lina v0.2 teknik olarak yükleniyor ve inference üretiyor; ancak kimlik sorusuna
dataset talimatı, secret sorusuna anlamsız metin ve dosya erişimi sorusuna ilgisiz
güvenlik sloganı verebiliyordu. Önceki recovery çalışması tokenizer, renderer ve
LoRA merge katmanlarının bu bozulmanın kaynağı olmadığını kanıtladı. Asıl risk,
600 kayıtlı eski dataset içindeki kategoriler arası sentetik slogan tekrarı ve
eğitim ezberlemesiydi.

Eski dataset yeni eğitim verisine aktarılmadı. “Kararı mevcut kanıta dayandırırım”,
“Sonucu kanıtla ayırırım” ve benzeri meta sloganlar yalnız negatif regression
örneği olarak korundu.

## Curated dataset stratejisi

`Lina Recovery Dataset v0.3`, kişisel veri içermeyen 96 sentetik fakat elle
küratörlü kayıttan oluşur. On altı kategorinin her birinde altı özgün prompt ve
cevap vardır. Yanıtların çoğu bir ila dört cümledir ve ilk cümle doğrudan cevabı
verir.

- Dataset: `data/examples/lina_recovery_v0.3.jsonl`
- Fingerprint: `sha256:6c48fde12484a4e17d3ea0c6b409aa37511e68c40095d4f45fc0166c1a205354`
- Stage 1 subset: 24 kayıt, dört kategori
- Stage 2 subset: 64 kayıt, on bir kategori
- Bağımsız holdout: 32 prompt
- Cross-category tekrarlanan 5-gram: 0
- Beklenmeyen Unicode script: 0
- Holdout/training leakage: 0 / 3.072 karşılaştırma

Stage 1 kimlik, secret güvenliği, dosya erişimi dürüstlüğü ve eksik bilgi
davranışlarına odaklandı. Stage 2 subset'i ayrıca LinaCore/Desktop, offline
kullanım, debugging ve doğal sohbet kayıtlarını içerir; fakat Stage 1 başarısız
olduğu için eğitimde kullanılmadı.

## Eğitim sözleşmesi

Tüm denemeler aynı local base model ve revision ile CPU üzerinde çalıştı:

- Base: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Dtype: float32
- Seed: 42
- Canonical Qwen chat template
- Assistant-only selective causal-LM loss
- Tek terminal EOS
- Grouped train/validation/test split: 16 / 4 / 4
- Validation-based best checkpoint ve early stopping
- LoRA dropout: 0.05
- Remote download: kapalı

## Curriculum Stage 1 sonuçları

| Deneme | LR | LoRA | Microstep / accumulation | Süre | Train loss | Validation loss | Best step | Holdout |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.5e-5 | r8 / a16 | 48 / 4 | 143,18 sn | 2,3819 | 3,0803 | 48 | 0/8 |
| 2 | 7.5e-5 | r8 / a16 | 96 / 2 | 129,74 sn | 2,0061 | 2,9203 | 64 | 0/8 |
| 3 | 1e-4 | r16 / a32 | 96 / 2 | 131,15 sn | 1,1884 | 2,8790 | 48 | 0/8 |

Üç denemede de kritik failure 8, yabancı script 0 ve generic slogan karışması
0 oldu. Validation iyileşmesi davranış kapısına dönüşmedi. Bu nedenle yalnız loss
değerine bakarak başarılı model ilan edilmedi.

Son denemeden bazı canlı çıktılar:

- Kimlik: “Ben Qwen, bu konuda yeterli bir bilgi yok. Ben bir AI asistanıyım...”
- Secret: “Evet, erişim anahtarını terminal çıktısına yapıştırma önerimim yok...”
- Dosya erişimi: “Hayır, belgeleri eklememem. Ben Qwen...”
- Eksik bilgi: “Bunu söyleyemem; uygulamayı teşhettiğinizi söyleyemem.”

Bu cevaplar keyword içerse bile doğal, anlamlı veya davranış açısından güvenilir
değildir. Hardened evaluator kararı insan incelemesiyle uyumludur.

Generated human-review paketleri Git tarafından ignore edilen `reports/`
klasöründe tutulur:

- `reports/lina-natural-stage1-review.md`
- `reports/lina-natural-stage1-2-review.md`
- `reports/lina-natural-stage1-3-review.md`

Her rapor prompt, beklenen davranış, model cevabı, otomatik değerlendirme, kritik
failure, dil/doğrudanlık durumu ve boş insan yorumu alanı içerir.

## Artifact doğrulaması

Üç canary ayrı dizinlere yazıldı; eski model artifact'lerinin üzerine yazılmadı.
Her canary için manifest `available/valid`, model card mevcut, diskten reload ve
inference başarılıdır. Checksum dosyalarındaki tüm girdiler yeniden hesaplandı ve
uyuşmazlık bulunmadı.

Son canary'nin best-step unmerged LoRA checkpoint'i ile merged artifact'i aynı
prompt üzerinde ayrıca karşılaştırıldı. İlk-token logit maksimum mutlak farkı
`5.2184e-05` ile `1e-4` toleransının altında kaldı; argmax, ilk 10 generation
token'ı, decoded çıktı, dtype ve tokenizer hash'i aynıydı. Merge equivalence
kapısı geçti.

Son canary:

- Artifact: `models/lina-0.5b-sft-v0.2.1-natural-stage1-3`
- Manifest: `models/lina-0.5b-sft-v0.2.1-natural-stage1-3/linacore-model.toml`
- Model card: `models/lina-0.5b-sft-v0.2.1-natural-stage1-3/MODEL_CARD.md`

Bu canary'ler araştırma çıktısıdır; Git'e eklenmez ve production-ready değildir.
Tam model üretilmediği için full holdout ve Stage 2 canlı 15-prompt kapısı
uygulanabilir değildir; merge eşdeğerliği yalnız son Stage 1 canary üzerinde
doğrulandı.

## Bilinen sınırlamalar ve sonraki yön

- 0.5B taban modelin Türkçe instruction-following kapasitesi bu dar dataset için
  yetersiz olabilir.
- 24 kayıt, kimlik ve güvenlik davranışlarını paraphrase holdout'a genellemek için
  yeterli olmadı.
- Daha düşük loss, doğal cevap kalitesini tek başına göstermedi.
- Sonraki çalışma daha fazla training yapmak yerine cevap çiftlerini insan
  değerlendirmesiyle güçlendirmeli ve base modelin aynı prompt pack'teki Türkçe
  yeterliliğini karar ölçütü olarak yeniden ele almalıdır.

Stage 1 başarı eşiği sağlanmadan Stage 2 veya full training çalıştırılmamalıdır.
