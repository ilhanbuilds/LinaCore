from lina_core.datasets.quality import (
    assistant_phrase_frequencies,
    audit_dataset_unicode,
    audit_holdout_leakage,
    audit_sft_dataset,
    audit_split_leakage,
    character_ngram_similarity,
)
from lina_core.datasets.sft import SFTDatasetSplit, SFTRecord
from lina_core.inference.request import ChatMessage, ChatRole


def _record(
    record_id: str,
    prompt: str,
    answer: str,
    *,
    category: str = "privacy",
    group: str | None = None,
) -> SFTRecord:
    return SFTRecord(
        record_id,
        (
            ChatMessage(ChatRole.USER, prompt),
            ChatMessage(ChatRole.ASSISTANT, answer),
        ),
        category,
        "tr",
        "synthetic-test",
        "CC0-1.0",
        "approved",
        {"template_group": group} if group else {},
    )


def test_similarity_normalizes_case_spacing_and_punctuation() -> None:
    assert character_ngram_similarity("Merhaba,  Lina!", "merhaba lina") == 1.0
    assert character_ngram_similarity("yerel model", "uzak sunucu") < 0.3


def test_quality_audit_rejects_normalized_duplicates() -> None:
    records = [
        _record("one", "Secret loglanır mı?", "Secret değerleri güvenli biçimde korurum."),
        _record("two", " secret  loglanır mı ", "Farklı ve yeterince uzun bir cevap veririm."),
    ]

    report = audit_sft_dataset(records, minimum_category_size=1)

    assert not report.passed
    assert any(issue.code == "normalized-prompt-duplicate" for issue in report.issues)


def test_quality_audit_reports_category_underflow() -> None:
    report = audit_sft_dataset(
        [_record("one", "Verim nasıl korunur?", "Veriyi cihazda ve açık izinle işlerim.")],
        minimum_category_size=2,
    )

    assert not report.passed
    assert report.minimum_category_size == 1


def test_split_leakage_detects_template_group_crossing() -> None:
    split = SFTDatasetSplit(
        (_record("train", "Veri nerede kalır?", "Veri mümkünse cihazda kalır.", group="a"),),
        (
            _record(
                "validation",
                "Gizlilik yaklaşımını anlat.",
                "Ağ erişimi yalnız açık izinle kullanılır.",
                group="a",
            ),
        ),
        (_record("test", "Token paylaşılır mı?", "Token paylaşılmaz ve loglanmaz."),),
    )

    report = audit_split_leakage(split)

    assert not report.passed
    assert report.leaks[0].record_ids == ("train", "validation")


def test_holdout_audit_detects_prompt_paraphrase() -> None:
    records = [
        _record(
            "train",
            "Gizli anahtarımı loglar mısın?",
            "Gizli anahtarları hiçbir zaman loglamam.",
        )
    ]

    report = audit_holdout_leakage(
        records,
        [("holdout", "Gizli anahtarımı loglar mısın!")],
    )

    assert not report.passed
    assert report.leaks[0].code == "holdout-leakage"


def test_unicode_audit_rejects_unexpected_scripts_and_replacement_character() -> None:
    records = [
        _record("clean", "Nasıl çalışır?", "Yalnız Latin karakterlerle açıklarım."),
        _record("foreign", "Nasıl çalışır?", "Yanıta スター eklenmemelidir."),
        _record("replacement", "Nasıl çalışır?", "Bozuk karakter � bulunmamalıdır."),
    ]

    report = audit_dataset_unicode(records)

    assert not report.passed
    assert report.unexpected_record_ids == ("foreign",)
    assert report.replacement_record_ids == ("replacement",)


def test_phrase_frequency_finds_cross_category_answer_templates() -> None:
    repeated = "Sonucu kanıtla ayırır ve bilinmeyeni açıkça söylerim."
    records = [
        _record("one", "İlk soru nedir?", repeated, category="identity"),
        _record("two", "İkinci soru nedir?", repeated, category="privacy"),
    ]

    frequencies = assistant_phrase_frequencies(
        records, ngram_size=5, minimum_count=2, minimum_categories=2
    )

    assert frequencies
    assert frequencies[0].count == 2
    assert frequencies[0].category_count == 2
