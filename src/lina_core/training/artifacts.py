import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from lina_core.core.exceptions import TrainingError


@dataclass(frozen=True, slots=True)
class ModelEvaluationSummary:
    schema_version: int
    decision: str
    target_pass_rate: float
    minimum_pass_rate: float
    base_model_score: float
    lina_v0_1_score: float
    lina_v0_2_score: float
    critical_failures: int
    generation_success_rate: float
    average_latency_seconds: float


def finalize_model_artifact(
    artifact_root: Path,
    comparison_report: Path,
    *,
    chat_template_policy: str = "required",
) -> ModelEvaluationSummary:
    root = artifact_root.resolve()
    try:
        metadata = json.loads((root / "training-metadata.json").read_text(encoding="utf-8"))
        comparison = json.loads(comparison_report.read_text(encoding="utf-8"))
        summaries = {item["model_id"]: item for item in comparison["summaries"]}
        base = summaries[str(metadata["base_model"])]
        previous = summaries["lina-135m-sft-v0.1"]
        current = summaries[str(metadata["model_id"])]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise TrainingError(f"Model artifact finalization verisi geçersiz: {exc}") from exc

    current_score = float(current["pass_rate"])
    critical_failures = int(current["critical_failures"])
    success = (
        current_score >= 0.75
        and current_score > float(base["pass_rate"])
        and current_score > float(previous["pass_rate"])
        and critical_failures == 0
    )
    summary = ModelEvaluationSummary(
        1,
        "SUCCESS" if success else "PARTIAL",
        0.75,
        0.60,
        float(base["pass_rate"]),
        float(previous["pass_rate"]),
        current_score,
        critical_failures,
        float(current["generation_success_rate"]),
        float(current["average_latency_seconds"]),
    )
    source_report = comparison_report.resolve()
    target_report = root / "base-vs-v0.1-vs-v0.2.json"
    if source_report != target_report:
        shutil.copyfile(source_report, target_report)
    (root / "evaluation-summary.json").write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = root / "linacore-model.toml"
    manifest_text = manifest.read_text(encoding="utf-8")
    manifest.write_text(
        manifest_text.replace(
            'chat_template_policy = "fallback"',
            f'chat_template_policy = "{chat_template_policy}"',
        ),
        encoding="utf-8",
    )
    (root / "MODEL_CARD.md").write_text(
        _model_card(metadata, summary),
        encoding="utf-8",
    )
    _write_checksums(root)
    return summary


def _model_card(metadata: dict[str, Any], summary: ModelEvaluationSummary) -> str:
    split = metadata["split"]
    lora = metadata["lora"]
    return f"""# {metadata["model_name"]}

## Model özeti

Bu model sıfırdan pretrained değildir. Apache-2.0 lisanslı
`{metadata["base_model"]}` revision `{metadata["base_revision"]}` üzerine LoRA tabanlı
supervised fine-tuning uygulanmış deneysel bir Lina türevidir. LoRA adapter'ı final
Transformers artifact'ine birleştirilmiştir.

Sprint kararı **{summary.decision}**: model base ve Lina v0.1'den daha iyi olsa da minimum
%60 davranış eşiğini veya sıfır kritik failure koşulunu karşılamamıştır. Üretim modeli değildir.

## Eğitim ve veri

- LinaCore sürümü: 0.3.0a0
- Yöntem: {metadata["training_method"]}
- Dataset: v{metadata["dataset_version"]}, {metadata["dataset_records"]} sentetik kayıt, CC0-1.0
- Kategori: 30
- Fingerprint: `{metadata["dataset_fingerprint"]}`
- Split: train {split["train"]}, validation {split["validation"]}, test {split["test"]}
- Device/dtype: {metadata["device"]} / {metadata["dtype"]}
- Süre: {metadata.get("duration_seconds", "training result metadata")}
- Step / best step: {metadata["training_steps"]} / {metadata["best_step"]}
- Final train loss: {metadata["final_training_loss"]}
- Validation loss: {metadata["validation_loss"]}
- Perplexity: {metadata["perplexity"]}

## LoRA

- r / alpha / dropout: {lora["r"]} / {lora["alpha"]} / {lora["dropout"]}
- Target modules: {", ".join(lora["target_modules"])}
- Trainable / toplam parametre: {metadata["trainable_parameters"]} / {metadata["total_parameters"]}
- Trainable oran: %{metadata["trainable_percentage"]:.3f}

## Bağımsız holdout değerlendirmesi

Aynı 90 prompt, temperature 0, max_new_tokens 64 ve seed 42 ile:

- Base model: %{summary.base_model_score * 100:.2f}
- Lina-135M-SFT-v0.1: %{summary.lina_v0_1_score * 100:.2f}
- Lina-0.5B-SFT-v0.2: %{summary.lina_v0_2_score * 100:.2f}
- Kritik failure: {summary.critical_failures}
- Generation success: %{summary.generation_success_rate * 100:.2f}
- Ortalama CPU latency: {summary.average_latency_seconds:.2f} sn

Rule sonucu insan değerlendirmesinin yerini tutmaz; raporda `human_review` alanı korunur.

## Amaçlanan ve amaçlanmayan kullanım

Yerel inference deneyi, LinaCore pipeline doğrulaması ve Türkçe davranış araştırması
içindir. Yüksek riskli karar, güvenlik garantisi, uzman tavsiyesi veya production otomasyonu
için kullanılmamalıdır.

## Gizlilik, güvenlik ve sınırlamalar

Dataset tamamen sentetiktir; kullanıcıya özel konuşma içermez. Secret ve kişisel veri modele
verilmemelidir. Model uydurabilir, promptu yanlış anlayabilir ve kritik güvenlik/dürüstlük
kurallarında başarısız olabilir. CPU inference yavaştır. GGUF export edilmemiştir.

## LinaCore ile çalıştırma

```powershell
python -m lina_core inference generate `
  --model lina-0.5b-sft-v0.2 `
  --prompt "Merhaba, sen kimsin?" `
  --max-new-tokens 96 `
  --temperature 0 `
  --seed 42
```
"""


def _write_checksums(root: Path) -> None:
    checksums = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(item for item in root.rglob("*") if item.is_file())
        if path.name != "checksums.json"
    }
    (root / "checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
