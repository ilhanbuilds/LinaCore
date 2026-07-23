import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from lina_core.datasets.sft import SFTDatasetSplit, SFTRecord
from lina_core.inference.request import ChatRole

_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[^\wçğıöşü]+", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DatasetQualityIssue:
    code: str
    severity: str
    message: str
    record_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetQualityReport:
    record_count: int
    category_count: int
    minimum_category_size: int
    maximum_category_size: int
    issues: tuple[DatasetQualityIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


@dataclass(frozen=True, slots=True)
class LeakageReport:
    comparisons: int
    leaks: tuple[DatasetQualityIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.leaks


def normalize_similarity_text(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold()
    return _SPACE.sub(" ", _WORD.sub(" ", folded)).strip()


def character_ngram_similarity(left: str, right: str, *, size: int = 3) -> float:
    left_normalized = normalize_similarity_text(left)
    right_normalized = normalize_similarity_text(right)
    if left_normalized == right_normalized:
        return 1.0
    left_ngrams = _ngrams(left_normalized, size)
    right_ngrams = _ngrams(right_normalized, size)
    if not left_ngrams or not right_ngrams:
        return 0.0
    return len(left_ngrams & right_ngrams) / len(left_ngrams | right_ngrams)


def audit_sft_dataset(
    records: Iterable[SFTRecord],
    *,
    minimum_category_size: int = 12,
    near_duplicate_threshold: float = 0.9,
) -> DatasetQualityReport:
    items = tuple(records)
    categories = Counter(item.category for item in items)
    issues: list[DatasetQualityIssue] = []
    if not categories:
        issues.append(DatasetQualityIssue("empty-dataset", "error", "Dataset boş."))
        return DatasetQualityReport(0, 0, 0, 0, tuple(issues))
    for category, count in sorted(categories.items()):
        if count < minimum_category_size:
            issues.append(
                DatasetQualityIssue(
                    "category-underflow",
                    "error",
                    f"{category} kategorisinde yalnız {count} kayıt var.",
                )
            )
    if max(categories.values()) > min(categories.values()) * 3:
        issues.append(
            DatasetQualityIssue(
                "category-imbalance",
                "warning",
                "En büyük kategori en küçük kategorinin üç katından fazla.",
            )
        )
    prompts: dict[str, str] = {}
    answers: dict[str, str] = {}
    template_counts: Counter[str] = Counter()
    for item in items:
        prompt = _last_message(item, ChatRole.USER)
        answer = _last_message(item, ChatRole.ASSISTANT)
        normalized_prompt = normalize_similarity_text(prompt)
        normalized_answer = normalize_similarity_text(answer)
        _record_duplicate(issues, prompts, normalized_prompt, item.record_id, "prompt")
        _record_duplicate(issues, answers, normalized_answer, item.record_id, "answer")
        words = normalized_answer.split()
        template_counts[" ".join(words[:6])] += 1
        if len(answer) < 24:
            issues.append(
                DatasetQualityIssue(
                    "assistant-too-short",
                    "error",
                    "Assistant cevabı 24 karakterden kısa.",
                    (item.record_id,),
                )
            )
        if len(answer) > 1_200:
            issues.append(
                DatasetQualityIssue(
                    "assistant-too-long",
                    "error",
                    "Assistant cevabı 1200 karakterden uzun.",
                    (item.record_id,),
                )
            )
    template_limit = max(8, round(len(items) * 0.04))
    for template, count in template_counts.items():
        if template and count > template_limit:
            issues.append(
                DatasetQualityIssue(
                    "answer-template-overuse",
                    "warning",
                    f"Cevap başlangıcı {count} kez kullanılmış: {template!r}.",
                )
            )
    grouped: dict[str, list[SFTRecord]] = defaultdict(list)
    for item in items:
        grouped[item.category].append(item)
    for category_items in grouped.values():
        for index, left in enumerate(category_items):
            left_prompt = _last_message(left, ChatRole.USER)
            for right in category_items[index + 1 :]:
                similarity = character_ngram_similarity(
                    left_prompt, _last_message(right, ChatRole.USER)
                )
                if similarity >= near_duplicate_threshold:
                    issues.append(
                        DatasetQualityIssue(
                            "near-duplicate-prompt",
                            "error",
                            f"Prompt benzerliği {similarity:.3f}.",
                            (left.record_id, right.record_id),
                        )
                    )
    return DatasetQualityReport(
        len(items),
        len(categories),
        min(categories.values()),
        max(categories.values()),
        tuple(issues),
    )


def audit_split_leakage(
    split: SFTDatasetSplit, *, near_duplicate_threshold: float = 0.82
) -> LeakageReport:
    named = (("train", split.train), ("validation", split.validation), ("test", split.test))
    comparisons = 0
    leaks: list[DatasetQualityIssue] = []
    for left_index, (left_name, left_items) in enumerate(named):
        for right_name, right_items in named[left_index + 1 :]:
            for left in left_items:
                for right in right_items:
                    comparisons += 1
                    same_group = bool(left.metadata.get("template_group")) and left.metadata.get(
                        "template_group"
                    ) == right.metadata.get("template_group")
                    similarity = character_ngram_similarity(
                        _last_message(left, ChatRole.USER), _last_message(right, ChatRole.USER)
                    )
                    if same_group or similarity >= near_duplicate_threshold:
                        leaks.append(
                            DatasetQualityIssue(
                                "split-leakage",
                                "error",
                                f"{left_name}/{right_name} prompt benzerliği {similarity:.3f}.",
                                (left.record_id, right.record_id),
                            )
                        )
    return LeakageReport(comparisons, tuple(leaks))


def audit_holdout_leakage(
    training: Iterable[SFTRecord],
    holdout_prompts: Iterable[tuple[str, str]],
    *,
    near_duplicate_threshold: float = 0.82,
) -> LeakageReport:
    comparisons = 0
    leaks: list[DatasetQualityIssue] = []
    training_items = tuple(training)
    for sample_id, prompt in holdout_prompts:
        for record in training_items:
            comparisons += 1
            similarity = character_ngram_similarity(
                prompt, _last_message(record, ChatRole.USER)
            )
            if similarity >= near_duplicate_threshold:
                leaks.append(
                    DatasetQualityIssue(
                        "holdout-leakage",
                        "error",
                        f"Holdout/training prompt benzerliği {similarity:.3f}.",
                        (sample_id, record.record_id),
                    )
                )
    return LeakageReport(comparisons, tuple(leaks))


def _ngrams(value: str, size: int) -> set[str]:
    padded = f"  {value}  "
    return {padded[index : index + size] for index in range(len(padded) - size + 1)}


def _last_message(record: SFTRecord, role: ChatRole) -> str:
    return next(message.content for message in reversed(record.messages) if message.role is role)


def _record_duplicate(
    issues: list[DatasetQualityIssue],
    seen: dict[str, str],
    value: str,
    record_id: str,
    kind: str,
) -> None:
    previous = seen.get(value)
    if previous is not None:
        issues.append(
            DatasetQualityIssue(
                f"normalized-{kind}-duplicate",
                "error",
                f"Normalize edilmiş {kind} içeriği yineleniyor.",
                (previous, record_id),
            )
        )
    else:
        seen[value] = record_id
