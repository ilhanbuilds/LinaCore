import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lina_core.core.exceptions import EvaluationError
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.inference.sampling import GenerationOptions
from lina_core.models.base import ModelProvider
from lina_core.security.limits import ResourceLimits
from lina_core.security.paths import safe_resolve


@dataclass(frozen=True, slots=True)
class BehaviorSample:
    sample_id: str
    prompt: str
    keywords: tuple[str, ...]
    category: str
    language: str


@dataclass(frozen=True, slots=True)
class ModelOutput:
    model_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    rule_passed: bool


@dataclass(frozen=True, slots=True)
class ComparisonItem:
    sample_id: str
    category: str
    prompt: str
    base: ModelOutput
    fine_tuned: ModelOutput
    human_review: str = "pending"


@dataclass(frozen=True, slots=True)
class ModelComparisonReport:
    schema_version: int
    base_model_id: str
    fine_tuned_model_id: str
    created_at: str
    generation_success_rate: float
    base_rule_pass_rate: float
    fine_tuned_rule_pass_rate: float
    average_base_latency: float
    average_fine_tuned_latency: float
    items: tuple[ComparisonItem, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def load_behavior_samples(path: Path) -> list[BehaviorSample]:
    samples: list[BehaviorSample] = []
    allowed = {"id", "prompt", "keywords", "category", "language"}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, 1):
            data = json.loads(line)
            if not isinstance(data, dict) or set(data) != allowed:
                raise EvaluationError(f"Eval satırı {number} strict şemaya uymuyor.")
            samples.append(
                BehaviorSample(
                    str(data["id"]),
                    str(data["prompt"]),
                    tuple(str(item) for item in data["keywords"]),
                    str(data["category"]),
                    str(data["language"]),
                )
            )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise EvaluationError(f"Eval dataset okunamadı: {exc}") from exc
    if not samples:
        raise EvaluationError("Eval dataset boş.")
    return samples


def compare_models(
    base: ModelProvider, fine_tuned: ModelProvider, samples: list[BehaviorSample]
) -> ModelComparisonReport:
    items: list[ComparisonItem] = []
    base.load()
    try:
        fine_tuned.load()
        try:
            for sample in samples:
                items.append(
                    ComparisonItem(
                        sample.sample_id,
                        sample.category,
                        sample.prompt,
                        _generate(base, sample),
                        _generate(fine_tuned, sample),
                    )
                )
        finally:
            fine_tuned.unload()
    finally:
        base.unload()
    base_passes = sum(item.base.rule_passed for item in items)
    tuned_passes = sum(item.fine_tuned.rule_passed for item in items)
    success = sum(bool(item.base.text) and bool(item.fine_tuned.text) for item in items)
    return ModelComparisonReport(
        1,
        base.metadata.model_id,
        fine_tuned.metadata.model_id,
        datetime.now(UTC).isoformat(),
        success / len(items),
        base_passes / len(items),
        tuned_passes / len(items),
        sum(item.base.latency_seconds for item in items) / len(items),
        sum(item.fine_tuned.latency_seconds for item in items) / len(items),
        tuple(items),
    )


def _generate(provider: ModelProvider, sample: BehaviorSample) -> ModelOutput:
    started = time.perf_counter()
    response = provider.generate(
        InferenceRequest(
            (ChatMessage(ChatRole.USER, sample.prompt),),
            GenerationOptions(max_new_tokens=32, temperature=0, seed=42),
            model_id=provider.metadata.model_id,
        )
    )
    folded = response.text.casefold()
    return ModelOutput(
        provider.metadata.model_id,
        response.text,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        time.perf_counter() - started,
        any(keyword.casefold() in folded for keyword in sample.keywords),
    )


def write_comparison_report(report: ModelComparisonReport, root: Path, filename: str) -> Path:
    target = safe_resolve(root, filename)
    payload = report.to_json() + "\n"
    if len(payload.encode()) > ResourceLimits().max_report_size:
        raise EvaluationError("Comparison report boyut sınırını aşıyor.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return target
