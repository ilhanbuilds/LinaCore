import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from lina_core.core.exceptions import EvaluationError
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.inference.sampling import GenerationOptions
from lina_core.models.base import ModelProvider
from lina_core.security.limits import ResourceLimits
from lina_core.security.paths import safe_resolve

_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[^\wçğıöşü]+", re.IGNORECASE)
_TURKISH_WORDS = frozenset(
    {
        "ama",
        "bir",
        "bu",
        "değil",
        "gerekir",
        "gerekli",
        "için",
        "ile",
        "olarak",
        "ve",
        "veya",
    }
)


@dataclass(frozen=True, slots=True)
class BehaviorSample:
    sample_id: str
    prompt: str
    category: str
    language: str
    required_concepts: tuple[tuple[str, ...], ...]
    normalized_answers: tuple[str, ...]
    forbidden_claims: tuple[str, ...]
    critical: bool
    human_review: str | None = None


@dataclass(frozen=True, slots=True)
class RuleResult:
    rule: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class BehaviorAssessment:
    passed: bool
    score: float
    critical_failure: bool
    rules: tuple[RuleResult, ...]


@dataclass(frozen=True, slots=True)
class BehaviorModelOutput:
    model_id: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    generation_success: bool
    assessment: BehaviorAssessment


@dataclass(frozen=True, slots=True)
class BehaviorComparisonItem:
    sample_id: str
    category: str
    prompt: str
    outputs: tuple[BehaviorModelOutput, ...]
    human_review: str | None


@dataclass(frozen=True, slots=True)
class BehaviorModelSummary:
    model_id: str
    passed: int
    total: int
    pass_rate: float
    critical_failures: int
    generation_success_rate: float
    average_latency_seconds: float


@dataclass(frozen=True, slots=True)
class BehaviorComparisonReport:
    schema_version: int
    created_at: str
    generation: dict[str, object]
    summaries: tuple[BehaviorModelSummary, ...]
    items: tuple[BehaviorComparisonItem, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def load_behavior_samples(path: Path) -> list[BehaviorSample]:
    allowed = {
        "id",
        "prompt",
        "category",
        "language",
        "required_concepts",
        "normalized_answers",
        "forbidden_claims",
        "critical",
        "human_review",
    }
    samples: list[BehaviorSample] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            data = json.loads(line)
            if not isinstance(data, dict) or set(data) != allowed:
                raise EvaluationError(
                    f"Behavior holdout satırı {line_number} strict şemaya uymuyor."
                )
            required = data["required_concepts"]
            if not isinstance(required, list) or not required:
                raise EvaluationError(f"Behavior holdout satırı {line_number} concept içermiyor.")
            samples.append(
                BehaviorSample(
                    str(data["id"]),
                    str(data["prompt"]),
                    str(data["category"]),
                    str(data["language"]),
                    tuple(tuple(str(term) for term in group) for group in required),
                    tuple(str(item) for item in data["normalized_answers"]),
                    tuple(str(item) for item in data["forbidden_claims"]),
                    bool(data["critical"]),
                    None if data["human_review"] is None else str(data["human_review"]),
                )
            )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise EvaluationError(f"Behavior holdout okunamadı: {exc}") from exc
    if not samples:
        raise EvaluationError("Behavior holdout boş.")
    return samples


def assess_behavior(sample: BehaviorSample, output: str) -> BehaviorAssessment:
    normalized_output = normalize_text(output)
    normalized_prompt = normalize_text(sample.prompt)
    words = normalized_output.split()
    rules = [
        RuleResult("non_empty", bool(normalized_output), "Çıktı boş olmamalı."),
        RuleResult(
            "question_copying",
            bool(normalized_output)
            and normalized_output != normalized_prompt
            and normalized_prompt not in normalized_output,
            "Prompt cevap olarak kopyalanmamalı.",
        ),
        RuleResult(
            "repetition",
            not _has_repeated_ngram(words),
            "Aynı üç kelimelik dizi üç kez yinelenmemeli.",
        ),
        RuleResult(
            "forbidden_claims",
            not any(
                normalize_text(claim) in normalized_output for claim in sample.forbidden_claims
            ),
            "Yasaklı iddia bulunmamalı.",
        ),
        RuleResult(
            "required_concepts",
            all(
                any(normalize_text(term) in normalized_output for term in alternatives)
                for alternatives in sample.required_concepts
            ),
            "Her gerekli kavram grubundan en az biri bulunmalı.",
        ),
    ]
    if sample.normalized_answers:
        rules.append(
            RuleResult(
                "normalized_match",
                normalized_output in {normalize_text(item) for item in sample.normalized_answers},
                "Normalize edilmiş beklenen cevaplardan biriyle eşleşmeli.",
            )
        )
    if sample.language == "tr":
        rules.append(
            RuleResult(
                "turkish_language",
                _looks_turkish(words, output),
                "Türkçe karakter veya yaygın Türkçe işlev sözcüğü bulunmalı.",
            )
        )
    passed_count = sum(rule.passed for rule in rules)
    passed = passed_count == len(rules)
    return BehaviorAssessment(
        passed,
        passed_count / len(rules),
        sample.critical and not passed,
        tuple(rules),
    )


def compare_behavior_models(
    providers: tuple[ModelProvider, ...],
    samples: list[BehaviorSample],
    *,
    max_new_tokens: int = 96,
    seed: int = 42,
) -> BehaviorComparisonReport:
    if not providers:
        raise EvaluationError("Değerlendirme en az bir model gerektirir.")
    by_sample: dict[str, list[BehaviorModelOutput]] = {sample.sample_id: [] for sample in samples}
    for provider in providers:
        provider.load()
        try:
            for sample in samples:
                by_sample[sample.sample_id].append(
                    _generate(provider, sample, max_new_tokens=max_new_tokens, seed=seed)
                )
        finally:
            provider.unload()
    items = tuple(
        BehaviorComparisonItem(
            sample.sample_id,
            sample.category,
            sample.prompt,
            tuple(by_sample[sample.sample_id]),
            sample.human_review,
        )
        for sample in samples
    )
    summaries = tuple(_summarize(provider.metadata.model_id, items) for provider in providers)
    return BehaviorComparisonReport(
        2,
        datetime.now(UTC).isoformat(),
        {
            "temperature": 0,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "stop_sequences": [],
        },
        summaries,
        items,
    )


def normalize_text(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return _SPACE.sub(" ", _WORD.sub(" ", folded)).strip()


def write_behavior_report(report: BehaviorComparisonReport, root: Path, filename: str) -> Path:
    target = safe_resolve(root, filename)
    payload = report.to_json() + "\n"
    if len(payload.encode("utf-8")) > ResourceLimits().max_report_size:
        raise EvaluationError("Behavior raporu boyut sınırını aşıyor.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return target


def _generate(
    provider: ModelProvider,
    sample: BehaviorSample,
    *,
    max_new_tokens: int,
    seed: int,
) -> BehaviorModelOutput:
    response = provider.generate(
        InferenceRequest(
            (ChatMessage(ChatRole.USER, sample.prompt),),
            GenerationOptions(max_new_tokens=max_new_tokens, temperature=0, seed=seed),
            model_id=provider.metadata.model_id,
        )
    )
    return BehaviorModelOutput(
        provider.metadata.model_id,
        response.text,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.latency,
        bool(response.text.strip()),
        assess_behavior(sample, response.text),
    )


def _summarize(model_id: str, items: tuple[BehaviorComparisonItem, ...]) -> BehaviorModelSummary:
    outputs = [
        next(output for output in item.outputs if output.model_id == model_id) for item in items
    ]
    passed = sum(output.assessment.passed for output in outputs)
    return BehaviorModelSummary(
        model_id,
        passed,
        len(outputs),
        passed / len(outputs),
        sum(output.assessment.critical_failure for output in outputs),
        sum(output.generation_success for output in outputs) / len(outputs),
        sum(output.latency_seconds for output in outputs) / len(outputs),
    )


def _has_repeated_ngram(words: list[str], *, size: int = 3) -> bool:
    if len(words) < size:
        return False
    ngrams = Counter(tuple(words[index : index + size]) for index in range(len(words) - size + 1))
    return max(ngrams.values(), default=0) >= 3


def _looks_turkish(words: list[str], original: str) -> bool:
    if any(character in original.casefold() for character in "çğıöşü"):
        return True
    return bool(_TURKISH_WORDS.intersection(words))
