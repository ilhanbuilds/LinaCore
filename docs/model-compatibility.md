# Model compatibility

Bu akış yalnız manuel entegrasyon doğrulaması içindir. Normal testler internet veya model
ağırlığı gerektirmez. İndirilen dosyalar `models/`, üretilen JSON raporlar `reports/`
altında kalır ve Git tarafından izlenmez.

## Rapor sözleşmesi

`ModelCompatibilityReport`; model kimliği, sürüm/revision, lisans, backend, cihaz, dtype,
genel sonuç, Python sürümü ve hata bilgisini taşır. Her kontrol kendi sonucu, süresi,
detayı ve sanitize edilmiş hatasıyla kaydedilir. Kontroller:

1. load
2. generation
3. streaming
4. cancellation
5. stop sequence
6. health
7. resources
8. unload

## Tekrarlama

Önce optional dependency’leri kurun:

```console
python -m pip install -e ".[inference-transformers,inference-torch]"
```

Model indirmek ağ erişimi ve açık onay gerektirir. Revision’ı tam commit SHA olarak pinleyin:

```console
python scripts/download_manual_model.py HuggingFaceTB/SmolLM2-135M ^
  --revision 93efa2f097d58c2a74874c7e644dbc9b0cee75a2 ^
  --output models/manual-smollm2-135m ^
  --allow-download
```

İndirilen dizine gerçek yolları göreli (`.`) tutan bir `linacore-model.toml` ekleyin. Sonra:

```console
python scripts/run_model_compatibility.py ^
  models/manual-smollm2-135m/linacore-model.toml ^
  --report smollm2-135m-compatibility.json ^
  --prompt "The quick brown fox"
```

Marker testi aynı manifestle açıkça etkinleştirilebilir:

```console
set LINA_CORE_MODEL_MANIFEST=models/manual-smollm2-135m/linacore-model.toml
python -m pytest -m model_integration
```

## Doğrulanmış sonuç

2026-07-22 koşusu:

| Alan | Değer |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M` |
| Revision | `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Lisans | Apache-2.0 |
| Ortam | Python 3.12.0, CPU, float32 |
| Sonuç | 8/8 kontrol başarılı |
| Load | 21.085 s |
| Generation | 0.531 s |
| Streaming | 0.504 s |
| Cancellation | 0.089 s |
| Stop sequence | 0.482 s |

Süreler tek bir geliştirme makinesindeki uyumluluk koşusudur; benchmark veya performans
garantisi değildir. Model kartındaki kullanım koşulları ve sınırlamalar LinaCore’dan
bağımsız olarak değerlendirilmelidir.
