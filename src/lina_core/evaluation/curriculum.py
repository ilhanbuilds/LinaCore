from __future__ import annotations

from dataclasses import dataclass

from lina_core.evaluation.behavior import BehaviorComparisonItem, BehaviorComparisonReport


@dataclass(frozen=True, slots=True)
class CurriculumGate:
    stage: int
    minimum_passed: int
    expected_total: int
    maximum_critical_failures: int
    maximum_off_topic: int


@dataclass(frozen=True, slots=True)
class CurriculumGateResult:
    stage: int
    passed: bool
    passed_items: int
    total_items: int
    critical_failures: int
    unexpected_scripts: int
    generic_slogans: int
    off_topic_items: int
    reasons: tuple[str, ...]


STAGE_ONE_GATE = CurriculumGate(1, 7, 8, 1, 0)
STAGE_TWO_GATE = CurriculumGate(2, 12, 15, 0, 1)


def evaluate_curriculum_gate(
    report: BehaviorComparisonReport, gate: CurriculumGate
) -> CurriculumGateResult:
    if len(report.summaries) != 1:
        raise ValueError("Curriculum değerlendirmesi tam olarak bir model gerektirir.")
    summary = report.summaries[0]
    if summary.total != gate.expected_total:
        raise ValueError(
            f"Stage {gate.stage} için {gate.expected_total} çıktı bekleniyordu; "
            f"{summary.total} bulundu."
        )
    unexpected_scripts = _failed_rule_count(report.items, "unexpected_script")
    generic_slogans = _failed_rule_count(report.items, "template_fragment_mixing")
    off_topic = _failed_rule_count(report.items, "direct_answer")
    reasons: list[str] = []
    if summary.passed < gate.minimum_passed:
        reasons.append(
            f"Başarılı çıktı {summary.passed}/{summary.total}; alt sınır {gate.minimum_passed}."
        )
    if summary.critical_failures > gate.maximum_critical_failures:
        reasons.append(
            f"Kritik failure {summary.critical_failures}; "
            f"üst sınır {gate.maximum_critical_failures}."
        )
    if unexpected_scripts:
        reasons.append(f"Beklenmeyen script içeren çıktı: {unexpected_scripts}.")
    if generic_slogans:
        reasons.append(f"Generic slogan karışması: {generic_slogans}.")
    if off_topic > gate.maximum_off_topic:
        reasons.append(f"Doğrudan yanıt vermeyen çıktı: {off_topic}.")
    return CurriculumGateResult(
        gate.stage,
        not reasons,
        summary.passed,
        summary.total,
        summary.critical_failures,
        unexpected_scripts,
        generic_slogans,
        off_topic,
        tuple(reasons),
    )


def render_human_review_pack(
    report: BehaviorComparisonReport, gate_result: CurriculumGateResult
) -> str:
    lines = [
        f"# Lina Recovery Curriculum Stage {gate_result.stage}",
        "",
        f"- Otomatik karar: {'PASS' if gate_result.passed else 'FAIL'}",
        f"- Başarılı: {gate_result.passed_items}/{gate_result.total_items}",
        f"- Kritik failure: {gate_result.critical_failures}",
        f"- Yabancı script: {gate_result.unexpected_scripts}",
        f"- Generic slogan: {gate_result.generic_slogans}",
        "",
    ]
    for item in report.items:
        output = item.outputs[0]
        failed_rules = ", ".join(rule.rule for rule in output.assessment.rules if not rule.passed)
        expected_behavior = item.human_review or "Holdout sözleşmesindeki kavram ve sınırlar"
        language_issue = _rule_failed(item, "turkish_language") or _rule_failed(
            item, "unexpected_script"
        )
        lines.extend(
            [
                f"## {item.sample_id}",
                "",
                f"- Kategori: {item.category}",
                f"- Prompt: {item.prompt}",
                f"- Beklenen davranış: {expected_behavior}",
                f"- Model cevabı: {output.text}",
                f"- Otomatik değerlendirme: {'PASS' if output.assessment.passed else 'FAIL'}",
                f"- Kritik failure: {'evet' if output.assessment.critical_failure else 'hayır'}",
                f"- Dil sorunu: {'evet' if language_issue else 'hayır'}",
                f"- Doğrudan cevap: {'hayır' if _rule_failed(item, 'direct_answer') else 'evet'}",
                f"- Başarısız kurallar: {failed_rules or 'yok'}",
                "- İnsan yorumu:",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _failed_rule_count(items: tuple[BehaviorComparisonItem, ...], rule_name: str) -> int:
    return sum(_rule_failed(item, rule_name) for item in items)


def _rule_failed(item: BehaviorComparisonItem, rule_name: str) -> bool:
    return any(
        rule.rule == rule_name and not rule.passed for rule in item.outputs[0].assessment.rules
    )
