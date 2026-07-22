# Model release süreci

Bir Lina model release adayı için dataset lisansı/fingerprint’i, pinli base revision, kullanılan
config fingerprint’i, training/validation raporu, base karşılaştırması, compatibility raporu,
model card ve checksum’lar birlikte doğrulanır. Ağırlıklar kaynak kod repository’sine eklenmez.

Model sürümü LinaCore paket sürümünden bağımsızdır. Örneğin bu sprintte LinaCore `0.3.0a0`,
deneysel model ise `Lina-135M-SFT-v0.1` kimliğini taşır. Public artifact dağıtımı yapılırsa
Apache-2.0 yükümlülükleri ve değişiklik bildirimi ayrıca gözden geçirilmelidir.
