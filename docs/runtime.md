# Runtime

Lifecycle `created`, `starting`, `ready`, `stopping`, `stopped` ve `failed` state’leri arasında kilitli geçiş yapar. Startup provider’ları çözer ve yükler; hata state’i `failed` olur. Shutdown aktif talepleri iptal eder, provider’ı unload eder ve seçimleri temizler.

Runtime request bazlı cancellation token, concurrency sınırı, inference/streaming, model switching recovery, health ve resource report sağlar. Health state, aktif provider/model/tokenizer, loaded durumu, request sayısı, son hata ve uptime içerir. Kalıcı network service bu sürümde yoktur.

