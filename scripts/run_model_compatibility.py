"""Run manual Transformers compatibility checks for a local model manifest."""

import argparse
from pathlib import Path

from lina_core.core.paths import ProjectPaths
from lina_core.models.compatibility import (
    TransformersCompatibilityRunner,
    write_compatibility_report,
)
from lina_core.models.local import provider_from_manifest
from lina_core.models.validation import load_model_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Local model compatibility raporu üretir.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--report", default="model-compatibility.json")
    parser.add_argument("--prompt", default="The quick brown fox")
    args = parser.parse_args()
    paths = ProjectPaths.discover()
    validation = load_model_manifest(args.manifest, paths.model_dir)
    if not validation.valid or validation.manifest is None:
        parser.error(validation.error or f"Model kullanılamıyor: {validation.status}")
    manifest = validation.manifest
    provider = provider_from_manifest(args.manifest, paths.model_dir)
    runner = TransformersCompatibilityRunner(
        provider,
        model_version=manifest.version,
        source_revision=manifest.source_revision,
        license_name=manifest.license,
    )
    report = runner.run(args.prompt)
    target = write_compatibility_report(report, paths.report_dir, args.report)
    print(report.to_json())
    print(f"Rapor: {target}")
    return 0 if report.status == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
