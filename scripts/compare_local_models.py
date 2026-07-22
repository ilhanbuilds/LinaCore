import argparse
import sys
from pathlib import Path

from lina_core.evaluation.comparison import (
    compare_models,
    load_behavior_samples,
    write_comparison_report,
)
from lina_core.models.local import provider_from_manifest


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="İki local LinaCore modelini karşılaştırır.")
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("fine_tuned_manifest", type=Path)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--eval", type=Path, default=Path("data/examples/lina_eval_v0.1.jsonl"))
    parser.add_argument("--report", default="model-comparison.json")
    args = parser.parse_args()
    root = args.model_root.resolve()
    report = compare_models(
        provider_from_manifest(args.base_manifest.resolve(), root),
        provider_from_manifest(args.fine_tuned_manifest.resolve(), root),
        load_behavior_samples(args.eval),
    )
    target = write_comparison_report(
        report, args.fine_tuned_manifest.resolve().parent / "reports", args.report
    )
    print(report.to_json())
    print(f"Rapor: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
