import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from lina_core.datasets.sft import SFTDatasetSplit, SFTRecord
from lina_core.inference.request import ChatRole

_SPACE = re.compile(r"\s+")
_WORD = re.compile(r"[^\wçğıöşü]+", re.IGNORECASE)
_TOKEN = re.compile(r"[\wçğıöşü]+", re.IGNORECASE)
_UNEXPECTED_SCRIPTS = ("CYRILLIC", "HIRAGANA", "KATAKANA", "HANGUL", "CJK")


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


@dataclass(frozen=True, slots=True)
class UnicodeAuditReport:
    script_counts: tuple[tuple[str, int], ...]
    unexpected_record_ids: tuple[str, ...]
    control_record_ids: tuple[str, ...]
    replacement_record_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not (
            self.unexpected_record_ids
            or self.control_record_ids
            or self.replacement_record_ids
        )


@dataclass(frozen=True, slots=True)
class PhraseFrequency:
    phrase: str
    count: int
    category_count: int
    record_ids: tuple[str, ...]


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
    unicode_report = audit_dataset_unicode(items)
    if unicode_report.unexpected_record_ids:
        issues.append(
            DatasetQualityIssue(
                "unexpected-unicode-script",
                "error",
                "Assistant cevaplarında beklenmeyen Unicode script bulundu.",
                unicode_report.unexpected_record_ids,
            )
        )
    if unicode_report.control_record_ids:
        issues.append(
            DatasetQualityIssue(
                "unicode-control-character",
                "error",
                "Assistant cevaplarında izin verilmeyen kontrol karakteri bulundu.",
                unicode_report.control_record_ids,
            )
        )
    if unicode_report.replacement_record_ids:
        issues.append(
            DatasetQualityIssue(
                "unicode-replacement-character",
                "error",
                "Assistant cevaplarında Unicode replacement character bulundu.",
                unicode_report.replacement_record_ids,
            )
        )
    phrase_limit = max(8, round(len(items) * 0.04))
    repeated_phrases = assistant_phrase_frequencies(
        items,
        ngram_size=5,
        minimum_count=phrase_limit + 1,
        minimum_categories=2,
    )
    for phrase_info in repeated_phrases[:12]:
        issues.append(
            DatasetQualityIssue(
                "cross-category-phrase-overuse",
                "warning",
                f"Cevap kalıbı {phrase_info.count} kez ve "
                f"{phrase_info.category_count} kategoride kullanılmış: "
                f"{phrase_info.phrase!r}.",
                phrase_info.record_ids[:8],
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


def audit_dataset_unicode(records: Iterable[SFTRecord]) -> UnicodeAuditReport:
    scripts: Counter[str] = Counter()
    unexpected: set[str] = set()
    controls: set[str] = set()
    replacements: set[str] = set()
    for record in records:
        answer = _last_message(record, ChatRole.ASSISTANT)
        for character in answer:
            name = unicodedata.name(character, "")
            script = _unicode_script(name, character)
            scripts[script] += 1
            if any(marker in name for marker in _UNEXPECTED_SCRIPTS):
                unexpected.add(record.record_id)
            if unicodedata.category(character).startswith("C"):
                controls.add(record.record_id)
            if character == "�":
                replacements.add(record.record_id)
    return UnicodeAuditReport(
        tuple(sorted(scripts.items())),
        tuple(sorted(unexpected)),
        tuple(sorted(controls)),
        tuple(sorted(replacements)),
    )


def assistant_phrase_frequencies(
    records: Iterable[SFTRecord],
    *,
    ngram_size: int = 5,
    minimum_count: int = 2,
    minimum_categories: int = 1,
) -> tuple[PhraseFrequency, ...]:
    counts: Counter[str] = Counter()
    categories: dict[str, set[str]] = defaultdict(set)
    record_ids: dict[str, list[str]] = defaultdict(list)
    for record in records:
        words = _TOKEN.findall(
            normalize_similarity_text(_last_message(record, ChatRole.ASSISTANT))
        )
        seen_in_record: set[str] = set()
        for index in range(len(words) - ngram_size + 1):
            phrase = " ".join(words[index : index + ngram_size])
            if phrase in seen_in_record:
                continue
            seen_in_record.add(phrase)
            counts[phrase] += 1
            categories[phrase].add(record.category)
            record_ids[phrase].append(record.record_id)
    frequencies = [
        PhraseFrequency(
            phrase,
            count,
            len(categories[phrase]),
            tuple(record_ids[phrase]),
        )
        for phrase, count in counts.items()
        if count >= minimum_count and len(categories[phrase]) >= minimum_categories
    ]
    return tuple(sorted(frequencies, key=lambda item: (-item.count, item.phrase)))


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


def _unicode_script(name: str, character: str) -> str:
    if "LATIN" in name:
        return "Latin"
    for marker in _UNEXPECTED_SCRIPTS:
        if marker in name:
            return marker.title()
    if character.isalpha():
        return "OtherLetter"
    return "NonLetter"


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
