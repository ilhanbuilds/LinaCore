import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from lina_core.core.config import load_config
from lina_core.core.exceptions import LinaCoreError
from lina_core.core.paths import ProjectPaths
from lina_core.core.version import __version__
from lina_core.datasets.validation import load_manifest
from lina_core.evaluation.models import EvaluationSample
from lina_core.evaluation.reports import write_report
from lina_core.evaluation.runner import EvaluationRunner
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.models.mock import MockModelProvider
from lina_core.runtime.service import RuntimeService
from lina_core.tokenizer.mock import MockTokenizerProvider
from lina_core.training.config import load_training_config
from lina_core.training.runner import TrainingRunner


def build_runtime(*, start: bool = False) -> RuntimeService:
    runtime = RuntimeService()
    runtime.register_model(MockModelProvider())
    runtime.register_tokenizer(MockTokenizerProvider())
    if start:
        runtime.startup("mock", "mock")
    return runtime


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lina-core", description="Lina Core foundation CLI")
    parser.add_argument("--debug", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    sub.add_parser("health")
    config = sub.add_parser("config").add_subparsers(dest="action", required=True)
    validate = config.add_parser("validate")
    validate.add_argument("path", nargs="?", type=Path)
    models = sub.add_parser("models").add_subparsers(dest="action", required=True)
    models.add_parser("list")
    runtime = sub.add_parser("runtime").add_subparsers(dest="action", required=True)
    runtime.add_parser("start")
    runtime.add_parser("smoke")
    dataset = sub.add_parser("dataset").add_subparsers(dest="action", required=True)
    dv = dataset.add_parser("validate")
    dv.add_argument("manifest", type=Path)
    training = sub.add_parser("training").add_subparsers(dest="action", required=True)
    td = training.add_parser("dry-run")
    td.add_argument("config", type=Path)
    evaluation = sub.add_parser("evaluation").add_subparsers(dest="action", required=True)
    er = evaluation.add_parser("run")
    er.add_argument("config", type=Path)
    return parser


def _evaluation(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    section = data["evaluation"]
    runtime = build_runtime(start=True)
    try:
        samples = [
            EvaluationSample(
                str(i),
                "smoke",
                item.get("language", "en"),
                prompt=item["prompt"],
                expected_output=item.get("expected_output"),
            )
            for i, item in enumerate(section.get("samples", []), 1)
        ]
        result = EvaluationRunner(runtime).run(samples, section.get("dataset_id", "adhoc"))
        report = write_report(
            result,
            ProjectPaths.discover().report_dir,
            section.get("report_name", "evaluation.json"),
        )
        return {
            "status": "completed",
            "samples": result.sample_count,
            "failures": result.failures,
            "report": str(report),
        }
    finally:
        runtime.shutdown()


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "version":
        return {"name": "Lina Core", "version": __version__}
    if args.command == "health":
        health = build_runtime().health()
        return {"status": health.status, "message": health.message}
    if args.command == "config":
        load_config(args.path)
        return {"status": "valid", "path": str(args.path) if args.path else "defaults"}
    if args.command == "models":
        return {
            "models": [
                {
                    "id": "mock",
                    "kind": "architecture-test-only",
                    "capabilities": ["text_generation", "chat", "streaming"],
                }
            ]
        }
    if args.command == "runtime" and args.action == "start":
        return {
            "status": "not-implemented",
            "message": "Bu özellik henüz uygulanmadı.",  # noqa: RUF001
            "network_service_started": False,
        }
    if args.command == "runtime":
        runtime = build_runtime(start=True)
        try:
            response = runtime.infer(InferenceRequest((ChatMessage(ChatRole.USER, "smoke"),)))
            return {
                "status": "ok",
                "provider": "mock",
                "response": response.text,
                "disclaimer": "architecture test only",
            }
        finally:
            runtime.shutdown()
    if args.command == "dataset":
        manifest = load_manifest(args.manifest)
        return {
            "status": "valid",
            "dataset_id": manifest.dataset_id,
            "records": manifest.record_count,
        }
    if args.command == "training":
        config = load_training_config(args.config)
        return TrainingRunner(ProjectPaths.discover().root).dry_run(config)
    if args.command == "evaluation":
        return _evaluation(args.config)
    raise LinaCoreError("Unsupported command")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        print(json.dumps(_execute(args), ensure_ascii=False, default=str))
        return 0
    except (LinaCoreError, OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        if args.debug:
            raise
        print(
            json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr
        )
        return 2
