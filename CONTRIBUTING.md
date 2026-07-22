# Katkıda bulunma

LinaCore’a katkılar memnuniyetle karşılanır. Proje; hafif temel kurulum, açık backend sözleşmeleri, local-first güvenlik ve doğrulanabilir davranış ilkelerini korur.

## Geliştirme ortamı

```console
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Transformers veya llama.cpp adapter’ı üzerinde çalışıyorsanız yalnız gerekli optional grubu kurun. Model ağırlığı, cache, dataset, credential, generated report veya kişisel dosya yolu commit etmeyin.

## Değişiklik kuralları

- Kod, identifier, dosya adı ve commit mesajları İngilizce olmalıdır.
- CLI kullanıcı mesajları ile README/docs metinleri doğal Türkçe olmalıdır.
- Public API değişikliklerini gerekçelendirin ve geriye uyumluluğu koruyun.
- Backend’e özgü ayrıntıları contract katmanına sızdırmayın.
- Yeni dependency için kullanım gerekçesi ve optional olma değerlendirmesi ekleyin.
- Test doubles varsayılan test koşusunda internet veya ağırlık gerektirmemelidir.

## Doğrulama

```console
python -m pytest -m "not model_integration"
python -m ruff check .
python -m mypy src/lina_core
python -m compileall -q src
python -m build
python scripts/audit_public_repository.py
```

Commit mesajlarında Conventional Commits kullanın. Pull request açıklamasında kapsamı, test kanıtını, bilinen sınırlamaları ve varsa model/dataset lisansını belirtin.

