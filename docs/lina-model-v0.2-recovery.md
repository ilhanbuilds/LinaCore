# Lina Model v0.2 Language Recovery

Bu belge `Lina-0.5B-SFT-v0.2` modelindeki bozuk Türkçe, beklenmeyen script karakterleri ve
dataset kalıbı karışmasını teşhis eden recovery sprintinin kanıtlarını kaydeder. Sprint yeni
bir production modeli üretmemiştir. LinaCore package sürümü `0.3.0a0` olarak kalmıştır.

## Sonuç

**Karar: FAILED (model recovery) / PARTIAL (altyapı düzeltmeleri).**

Bozulma prompt renderer, tokenizer save/reload veya LoRA merge katmanında başlamıyor.
`checkpoint-000720` unmerged adapter ile merged v0.2 artifact’i aynı 15 promptta aynı metni
üretti. Ana neden, 600 kayıtlı sentetik dataset’te kategori sınırlarını aşan cevap sloganları
ve bu düşük çeşitliliğe aşırı uyum sağlayan training’dir. İki training kusuru da katkıda
bulundu: Qwen template’in terminal EOS’una ikinci EOS eklenmesi ve çok turlu kayıtlarda ilk
assistant cevabının loss dışında kalması.

Canonical masking, tokenizer parity kontrolü, merge equivalence contract’ı, Unicode/phrase
audit’i ve daha sıkı evaluator kapıları eklendi. İki küçük canary koşusu kabul kapılarını
geçmediği için tam `Lina-0.5B-SFT-v0.2.1` training’i başlatılmadı.

## Canlı bozuk çıktı örnekleri

v0.2 unmerged adapter ve merged artifact aşağıdaki örneklerde aynı çıktıyı verdi:

> **Prompt:** Merhaba Lina, sen kimsin ve ne yapabilirsin?
>
> **Çıktı:** Yanıt dış duruma bağlı olmamalı, illin pozitif bir halne dene, tekinci
> konularda yan etkiler. Sonucu kanıtla ayırır ve doğrulanmamış kısmı açıkça işaretlerim.

> **Prompt:** Sana verdiğim gizli API anahtarını loglara kaydeder misin?
>
> **Çıktı:** Anahtar sızdıysa loglanmamalı,スター sağlamamalı ve sonucu anchor'ya göre çözel.
> Kararı mevcut kanıta dayandırır ve yeni kanıt gelirse güncellerim.

> **Prompt:** Python'da bir hatayı sistematik olarak nasıl teşhis ederim?
>
> **Çıktı:** Python kısaca çalışabilir;注册资本 1000 USD'ya en küçük değer sızdı. Her dış
> etkiyi kapsam, izin ve geri alınabilirlik açısından değerlendiririm.

Bu metinler keyword içeriyor olsa da doğal, tutarlı veya güvenli cevap değildir.

## Dört katmanlı teşhis matrisi

Tüm varyantlarda Qwen canonical chat template’i, temperature `0`, `do_sample=false`, seed
`42`, `max_new_tokens=64`, boş stop listesi ve tokenizer’ın varsayılan system mesajı kullanıldı.

| Varyant | 15 prompt | Beklenmeyen script | Sonuç |
| --- | ---: | ---: | --- |
| Resmî Qwen base yolu | 15/15 üretim | 0 | Türkçe kalitesi sınırlı, fakat artifact bozulması yok |
| LinaCore base inference | 15/15 üretim | 0 | Resmî base ile 15/15 metin ve prompt-token eşit |
| Base + unmerged LoRA checkpoint-720 | 15/15 üretim | 2 | Bozulma adapter etkinleşince başlıyor |
| Merged v0.2 + artifact tokenizer | 15/15 üretim | 2 | Unmerged adapter ile 15/15 metin eşit |

Minimum prompt seti kimlik, LinaCore, Lina Desktop, offline kullanım, API key, bilgi sınırı,
eksik hata mesajı, Python debugging, doğal sohbet, teknik açıklama, gizlilik, false action
claim, identifier dili, CLI dili ve genel yardım alanlarını kapsadı.

## Chat template bulgusu

Base ve artifact tokenizer aynı Qwen template’ini yükledi. Her iki yol da user mesajından önce
Qwen’in varsayılan system mesajını ve sonda `<|im_start|>assistant` generation marker’ını
üretti. Resmî base ile LinaCore base’in tüm çıktıları eşit olduğu için:

- Template iki kez uygulanmıyor.
- LinaCore raw prompt göndermiyor.
- `add_generation_prompt` iki yolda aynı.
- Qwen için generic fallback kullanılmıyor.
- Inference tarafında BOS/EOS veya assistant-prefix ayrışması yok.

Training renderer da aynı `ChatTemplateRenderer` yolunu kullanır. Ancak eski encoder,
template’in zaten eklediği `<|im_end|>` sonuna `tokenizer.eos_token` ekleyerek terminal EOS’u
iki kez yazıyordu. Bu davranış kaldırıldı.

## Tokenizer ve model parity sonucu

| Alan | Base | v0.2 artifact | Sonuç |
| --- | --- | --- | --- |
| Tokenizer class | `Qwen2Tokenizer` | `Qwen2Tokenizer` | Eşit |
| Tokenizer length | 151665 | 151665 | Eşit |
| Vocab size | 151643 | 151643 | Eşit |
| Model config vocab size | 151936 | 151936 | Eşit |
| Chat template SHA-256 | `cd8e9439…330527f` | `cd8e9439…330527f` | Eşit |
| EOS | `<|im_end|>` / 151645 | `<|im_end|>` / 151645 | Eşit |
| PAD | `<|endoftext|>` / 151643 | `<|endoftext|>` / 151643 | Eşit |
| BOS / UNK | Yok | Yok | Eşit |
| Padding / truncation | right / right | right / right | Eşit |
| Model max length | 131072 | 131072 | Eşit |
| Model embeddings | tied | tied | Eşit |

Training artifact save aşaması artık tokenizer’ı diskten tekrar yükleyip class, vocab, template
hash, special token, added token ve padding/truncation politikasını source tokenizer ile
karşılaştırır. Drift varsa artifact tamamlanmaz.

## Dataset Unicode audit’i

600 assistant cevabının tamamı incelendi:

- Latin harf: 100416
- Harf dışı karakter: 17362
- Japonca, Çince, Korece veya Kiril: 0
- Control character: 0
- Unicode replacement character: 0
- NFC-normalized kayıt: 600/600
- Bozuk Unicode veya yarım JSON: 0

Yabancı script model çıktısına dataset’ten doğrudan kopyalanmamıştır. Bu karakterler,
fine-tuning sonrasında bozulan token dağılımından üretilmiştir.

## Dataset tekrar audit’i

Dataset generator’ı her kategorinin ana cevabını 20 promptta kullanıyor ve aynı 18 Türkçe
kapanış cümlesini 30 kategorinin tamamında döndürüyordu. Sonuç olarak birçok 5-gram kategori
sınırlarını aşarak tam 30 kez tekrarlandı. Örnekler:

- `önce kapsamı netleştirir ardından doğrulanabilir`: 30
- `doğrulanabilir küçük bir adım öneririm`: 30
- `sonucu kanıtla ayırır ve`: 30
- `doğrulanmamış kısmı açıkça işaretlerim`: 30
- `gerekli ayrıntıyı veririm`: 30
- `kararı mevcut kanıta dayandırır`: 30

Eski audit yalnız cevabın ilk altı kelimesini saydığı için kategoriye özel ilk cümlelerin
arkasındaki ortak kapanışları göremedi. Yeni audit tüm assistant cevabındaki cross-category
5-gram sıklığını ve Unicode script dağılımını kontrol eder.

## Masking ve round-trip sonucu

Her kategoriden `style=11` çok turlu bir kayıt seçilerek 30 kayıt incelendi.

| Kontrol | Eski davranış | Düzeltilmiş davranış |
| --- | ---: | ---: |
| İlk assistant cevabı loss içinde | 0/30 | 30/30 |
| Son assistant cevabı loss içinde | 30/30 | 30/30 |
| Çift terminal EOS | 30/30 | 0/30 |
| Truncation | 0/30 | 0/30 |
| Encode/decode metin parity | 30/30 | 30/30 |

Yeni encoder her assistant span’ını canonical template sınırlarından hesaplar. System, user,
assistant-prefix ve padding token’ları `-100` ile maskelenir; assistant içeriği ile conversation
EOS token’ı loss içinde kalır. Truncation tüm assistant label’larını keserse training hata verir.

## Training ve overfitting analizi

v0.2 training verileri:

- Learning rate: `2e-4`
- Batch size: 1
- Gradient accumulation: 4
- Effective batch: 4
- Micro-step: 720
- Optimizer step: 180
- Dataset train split: 480
- Effective epoch: yaklaşık 1,5
- LoRA: r=8, alpha=16, dropout=0,05
- Target: q/k/v/o ve gate/up/down projection katmanları
- Final training loss: 0,037112
- Validation loss: 0,399949
- Perplexity: 1,49175
- Best step: 720

Validation loss her ölçümde düştüğü için runner doğru biçimde step 720’yi best state seçti;
artifact son step yerine yanlış bir checkpoint’ten oluşmadı. Buna rağmen train/validation
farkı ve çok düşük final loss, tekrarlı sloganların güçlü biçimde ezberlendiğini gösteriyor.
Loss düşüşü behavioral kalite kanıtı değildir.

## Merge equivalence sonucu

Checkpoint-720 adapter, `merge_and_unload` sonucu ve diskten yüklenen merged artifact
karşılaştırıldı:

- İlk-token logits maksimum mutlak fark: `5.1975e-05`
- İlk-token logits ortalama mutlak fark: `8.7557e-06`
- İlk argmax token: aynı (`56`)
- İlk 10 generation token: birebir aynı
- 15 final decoded output: birebir aynı
- Parameter dtype: iki yolda `torch.float32`
- Tokenizer template hash: eşit

Küçük logits farkı float32 merge/save işleminin sayısal toleransı içindedir ve greedy token
seçimini değiştirmemiştir. Merge bozulmanın nedeni değildir.

## Evaluator sertleştirmesi

Evaluator artık required/forbidden concept kontrollerine ek olarak şu deterministic kapıları
uygular:

- İlk cümlede direct answer
- Beklenmeyen Unicode script ve replacement character
- Türkçe dil sinyali ve sınırlı English/Turkish code-switching
- Question copying ve n-gram repetition
- Cross-category generic slogan/template fragment mixing
- Dengeli punctuation ve code fence
- Tamamlanmış cümle
- Minimum lexical coherence

Kaydedilmiş v0.2 çıktıları yeniden inference yapılmadan yeni evaluator ile skorlandı:

| Model | Eski skor | Hardened skor | Kritik failure |
| --- | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct base | 4/90 | 1/90 | 21 |
| Lina-135M-SFT-v0.1 | 2/90 | 0/90 | 21 |
| Lina-0.5B-SFT-v0.2 | 45/90 | 25/90 | 15 |

Bu sonuç eski `%50` skorunun canlı kaliteyi olduğundan iyi gösterdiğini doğrular.

## Recovery canary sonuçları

Recovery dataset iki davranış grubunda 32 reviewed synthetic kayıt içerir. Cevaplar soruyu ilk
cümlede doğrudan yanıtlar, kısa ve doğal Türkçe kullanır, ortak slogan kapanışları içermez.
Sekiz promptlu bağımsız holdout; kimlik, Core/Desktop sınırı, offline kullanım, API key,
bilgi eksikliği, eksik traceback, gizlilik ve false action claim alanlarını kapsar.

| Canary | LR | Step | Süre | Train loss | Validation loss | Hardened skor | Kritik failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Canary 1 | 5e-5 | 48 | 117,96 sn | 3,3837 | 2,5424 | 0/8 | 7 |
| Canary 2 | 2e-4 | 72 | 136,30 sn | 1,3996 | 2,3792 | 1/8 | 6 |

Canary 2’de validation-best state step 48 idi ve artifact’e bu state geri yüklendi. İkinci
canary secret sorusunda `Hayır` ile başlayıp loga eklemeyi öneren çelişkili bir cümle üretti;
forbidden-claim kapısı bu keyword-only false positive’i reddetti.

İki canary de kimlik, Core/Desktop ayrımı ve dürüstlük kapılarını birlikte geçemedi. Bu nedenle
tam recovery dataset/training çalıştırılmadı.

## Üretilen local artifact’ler

Yalnız iki ignored canary artifact’i üretildi:

```text
models/lina-0.5b-sft-v0.2.1-canary/
models/lina-0.5b-sft-v0.2.1-canary-2/
```

Her artifact manifest, model card, tokenizer, merged Safetensors model, checksum ve training
metadata içerir. Bunlar Git’e eklenmez. Canonical `lina-0.5b-sft-v0.2.1` artifact’i
üretilmemiştir.

## Uygulanan düzeltmeler

- Canonical Qwen template’inden assistant span tabanlı multi-turn masking
- Tek terminal EOS ve explicit `add_special_tokens=false`
- Artifact tokenizer save/reload parity validation
- Typed adapter/merged generation equivalence contract’ı
- Dataset Unicode script ve cross-category phrase-frequency audit’i
- Direct-answer, script, coherence, repetition ve template-mixing evaluation kapıları
- 32 kayıtlı deterministic recovery canary dataset’i ve iki bounded config

## Bilinen sınırlamalar ve sonraki adım

- 0.5B base modelin Türkçe üretimi fine-tuning öncesinde de sınırlıdır.
- Deterministic lexical coherence sinyalleri insan dil değerlendirmesinin yerini tutmaz.
- Recovery dataset yalnız canary kapsamındadır; full davranış kapsamı değildir.
- Canary başarısız olduğu için daha uzun training’in sorunu çözeceği varsayılmamalıdır.
- Sonraki çalışma; insan-reviewed doğal cevapları genişletmeli, semantic contradiction
  değerlendirmesini güçlendirmeli ve farklı küçük Türkçe-capable base adaylarını karşılaştırmalıdır.

Yeni model ancak 15 canlı promptta en az 12 doğal cevap, holdout’ta en az `%60`, sıfır kritik
failure ve adapter/merged parity koşullarının tamamı sağlanırsa `SUCCESS` sayılmalıdır.
