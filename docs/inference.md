# Inference

Chat message listesi önce `ChatTemplateRenderer` ile prompt’a çevrilir. Tokenizer chat template’i varsa kullanılır; yoksa fallback yalnız açık policy ile etkinleşir. Runtime içine Lina kişiliği veya Türkçe cevap zorlayan system prompt gömülmez.

Context budgeting prompt/output/reserved token sayılarını raporlar. Varsayılan overflow politikası controlled `ContextLimitError` üretir. Optional oldest truncation system ve son user mesajını, ayrıca conversation turn bütünlüğünü korur.

Generation greedy/sampling ayrımını açıkça yapar, seed uygular, stop sequence işler, token usage ve latency ölçer. Streaming `started`, sıralı token event’leri ve tek terminal event üretir. Cancellation request ID ile thread-safe yönetilir; backend worker hataları raw exception olarak dışarı çıkmaz.

`Lina-135M-SFT-v0.1` local manifesti mevcutsa aynı Transformers yolu üzerinden yüklenir.
Deterministik değerlendirme `temperature=0` ve `seed=42` kullanır; demo sampling sonucu test
kanıtı olarak sunulmaz.
