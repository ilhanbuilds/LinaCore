# Güvenlik politikası

## Desteklenen sürümler

LinaCore pre-alpha aşamasındadır. Güvenlik düzeltmeleri yalnız en güncel geliştirme dalında hazırlanır; henüz uzun süreli destek garantisi yoktur.

## Açık bildirme

Bir güvenlik açığını public issue olarak paylaşmayın. GitHub repository sahibine özel kanaldan ulaşın veya GitHub Security Advisory özelliği etkinse private vulnerability report kullanın. Rapora etkiyi, tekrar üretme adımlarını ve güvenli bir örneği ekleyin; gerçek credential ya da kişisel veri göndermeyin.

## Güven sınırları

LinaCore model dosyalarını güvenilir kabul etmez. Varsayılan politika local-only çalışır, `trust_remote_code=false` uygular ve model root dışına çıkan path’leri reddeder. Buna rağmen model ağırlıkları, native inference dependency’leri ve third-party tokenizer kodları ayrıca incelenmelidir.

Repository’ye secret, token, credential, private key, gerçek kullanıcı verisi, model ağırlığı veya private dataset eklemeyin. Yanlışlıkla secret commit edilirse yalnız dosyayı silmek yeterli değildir; credential derhâl rotate edilmeli ve Git geçmişi repository sahibiyle koordineli biçimde temizlenmelidir.

