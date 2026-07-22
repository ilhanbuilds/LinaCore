# Donanım

CPU her zaman güvenli fallback olarak raporlanır. CUDA device’ları Torch mevcutsa index, ad, memory ve compute capability ile tespit edilir. MPS yalnız platform gerçekten kullanılabilir olduğunu bildirirse eklenir. DirectML gelecekteki backend capability’si olarak planlanmıştır.

`auto` önerilen accelerator’ı, aksi halde CPU’yu seçer. `cpu`, `cuda` ve `cuda:N` açık politikalardır. CPU üzerinde float16 reddedilir; auto dtype CPU’da float32, accelerator’da float16 seçer. Büyük model/CPU kombinasyonu için warning sözleşmesi vardır. Memory bilgisi bulunamadığında uydurulmaz.
