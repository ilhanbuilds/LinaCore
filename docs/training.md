# Training

Training bu sürümde yalnızca dry-run’dır. Typed config ve pozitif hyperparameter değerleri doğrulanır; manifest/output path’leri workspace içinde çözülür ve environment raporu üretilir. GPU ayrılmaz, model yüklenmez ve training loop başlatılmaz.

Gelecekte reproducibility için seed, immutable manifest, environment capture, checksummed input ve metric sürümleri kaydedilmelidir. Checkpoint retention sınırlandırılmalı; resumable state ile exported weight ayrılmalı ve her artefaktın lisansı ayrıca incelenmelidir.

