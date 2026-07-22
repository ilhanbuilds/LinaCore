# Modeller

Her local model `<model-root>/<model>/linacore-model.toml` yapısında tutulur. Manifest backend, architecture, göreli model/tokenizer yolu, source/revision, license, parameter count, context length, device/dtype, capability, chat template policy ve optional checksum içerir.

Discovery yalnız configured root altında manifest arar; model yüklemez veya ağırlık okumaz. Duplicate ID ve invalid manifest ayrı raporlanır. Model yolu root dışına çıkamaz. Örnek manifest `example=true` olduğu için gerçek model olarak yüklenmez. Kullanıcı her modelin lisansını ve checksum/provenance bilgisini ayrıca doğrulamalıdır.

