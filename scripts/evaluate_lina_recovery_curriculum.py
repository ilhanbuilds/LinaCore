"""Evaluate one local Lina recovery artifact and write an ignored human-review pack."""

from __future__ import annotations

import argparse
from pathlib import Path

from lina_core.evaluation.behavior import compare_behavior_models, load_behavior_samples
from lina_core.evaluation.curriculum import (
    STAGE_ONE_GATE,
    STAGE_TWO_GATE,
    evaluate_curriculum_gate,
    render_human_review_pack,
)
from lina_core.models.local import provider_from_manifest

STAGE_ONE_CATEGORIES = {"identity", "secret_safety", "file_access_honesty", "missing_information"}


def select_samples(path: Path, stage: int):  # type: ignore[no-untyped-def]
    samples = load_behavior_samples(path)
    if stage == 1:
        return [sample for sample in samples if sample.category in STAGE_ONE_CATEGORIES]
    selected = [sample for sample in samples if sample.category in STAGE_ONE_CATEGORIES]
    selected.extend(sample for sample in samples if sample.category not in STAGE_ONE_CATEGORIES)
    return selected[:15]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument(
        "--holdout", type=Path, default=Path("data/evaluation/lina_recovery_holdout_v0.3.jsonl")
    )
    parser.add_argument("--stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--review-output", required=True, type=Path)
    args = parser.parse_args()
    provider = provider_from_manifest(args.manifest, args.model_root)
    report = compare_behavior_models((provider,), select_samples(args.holdout, args.stage))
    result = evaluate_curriculum_gate(report, STAGE_ONE_GATE if args.stage == 1 else STAGE_TWO_GATE)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(report.to_json() + "\n", encoding="utf-8")
    args.review_output.write_text(render_human_review_pack(report, result), encoding="utf-8")
    print(render_human_review_pack(report, result).split("\n\n", 1)[0])
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
