import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

from lina_core.backends.llama_cpp.availability import llama_cpp_availability
from lina_core.backends.transformers.availability import transformers_availability
from lina_core.core.config import load_config
from lina_core.core.exceptions import LinaCoreError
from lina_core.core.paths import ProjectPaths
from lina_core.core.version import RELEASE_NAME, __version__
from lina_core.datasets.validation import load_manifest
from lina_core.diagnostics.environment import environment_report
from lina_core.diagnostics.hardware import detect_devices
from lina_core.diagnostics.report import diagnostics_text
from lina_core.evaluation.models import EvaluationSample
from lina_core.evaluation.reports import write_report
from lina_core.evaluation.runner import EvaluationRunner
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.inference.sampling import GenerationOptions
from lina_core.models.discovery import DiscoveredModel, discover_models
from lina_core.models.local import provider_from_manifest
from lina_core.models.mock import MockModelProvider
from lina_core.models.validation import load_model_manifest
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
    parser = argparse.ArgumentParser(prog="lina-core", description="Lina Core yerel inference CLI")
    parser.add_argument("--debug", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version")
    sub.add_parser("health")
    diagnostics = sub.add_parser("diagnostics")
    diagnostics.add_argument("--json", action="store_true", dest="as_json")
    backends = sub.add_parser("backends").add_subparsers(dest="action", required=True)
    backends.add_parser("list")
    devices = sub.add_parser("devices").add_subparsers(dest="action", required=True)
    devices.add_parser("list")
    config = sub.add_parser("config").add_subparsers(dest="action", required=True)
    validate = config.add_parser("validate")
    validate.add_argument("path", nargs="?", type=Path)
    models = sub.add_parser("models").add_subparsers(dest="action", required=True)
    models.add_parser("discover")
    models.add_parser("list")
    model_validate = models.add_parser("validate")
    model_validate.add_argument("target")
    runtime = sub.add_parser("runtime").add_subparsers(dest="action", required=True)
    runtime.add_parser("start")
    runtime.add_parser("smoke")
    runtime_load = runtime.add_parser("load")
    runtime_load.add_argument("model_id")
    runtime.add_parser("unload")
    inference = sub.add_parser("inference").add_subparsers(dest="action", required=True)
    for action in ("generate", "stream"):
        command = inference.add_parser(action)
        command.add_argument("--prompt", required=True)
        command.add_argument("--model", required=True)
        command.add_argument("--max-new-tokens", type=int, default=64)
        command.add_argument("--temperature", type=float, default=0.0)
        command.add_argument("--device", default=None)
        command.add_argument("--seed", type=int, default=42)
    benchmark = sub.add_parser("benchmark").add_subparsers(dest="action", required=True)
    benchmark.add_parser("run")
    dataset = sub.add_parser("dataset").add_subparsers(dest="action", required=True)
    dataset_validate = dataset.add_parser("validate")
    dataset_validate.add_argument("manifest", type=Path)
    training = sub.add_parser("training").add_subparsers(dest="action", required=True)
    training_dry_run = training.add_parser("dry-run")
    training_dry_run.add_argument("config", type=Path)
    evaluation = sub.add_parser("evaluation").add_subparsers(dest="action", required=True)
    evaluation_run = evaluation.add_parser("run")
    evaluation_run.add_argument("config", type=Path)
    return parser


def _evaluation(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    section = data["evaluation"]
    runtime = build_runtime(start=True)
    try:
        samples = [
            EvaluationSample(
                str(index),
                "smoke",
                item.get("language", "en"),
                prompt=item["prompt"],
                expected_output=item.get("expected_output"),
            )
            for index, item in enumerate(section.get("samples", []), 1)
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


def _model_root() -> Path:
    return ProjectPaths.discover().model_dir


def _model_dict(item: DiscoveredModel) -> dict[str, Any]:
    return {
        "model_id": item.model_id,
        "display_name": item.display_name,
        "backend": item.backend,
        "format": item.model_format,
        "status": item.status,
        "device": item.device,
        "license": item.license,
        "manifest": str(item.manifest_path),
    }


def _find_model(model_id: str) -> DiscoveredModel:
    result = discover_models(_model_root())
    match = next((item for item in result.models if item.model_id == model_id), None)
    if match is None:
        raise LinaCoreError(f"Model bulunamadı: {model_id}")
    if not match.validation.valid or match.status != "available":
        raise LinaCoreError(f"Model kullanılamıyor: {match.status}")
    return match


def _real_inference(args: argparse.Namespace) -> dict[str, Any]:
    model = _find_model(args.model)
    provider = provider_from_manifest(model.manifest_path, _model_root())
    if args.device:
        raise LinaCoreError("CLI device override henüz manifest politikasını değiştiremez.")
    provider.load()
    try:
        request = InferenceRequest(
            (ChatMessage(ChatRole.USER, args.prompt),),
            GenerationOptions(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=args.seed,
            ),
            model_id=args.model,
            stream=args.action == "stream",
        )
        if args.action == "generate":
            response = provider.generate(request)
            return {
                "status": "completed",
                "model_id": response.model_id,
                "text": response.text,
                "finish_reason": response.finish_reason,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "latency": response.latency,
            }
        events = list(provider.stream_generate(request))
        return {
            "status": events[-1].type if events else "error",
            "events": [
                {"type": event.type, "text": event.text, "index": event.index} for event in events
            ],
        }
    finally:
        provider.unload()


def _execute(args: argparse.Namespace) -> dict[str, Any] | str:
    if args.command == "version":
        return {"name": "Lina Core", "version": __version__, "release": RELEASE_NAME}
    if args.command == "health":
        health = build_runtime().health()
        return {"status": health.status, "message": health.message, "details": health.details}
    if args.command == "diagnostics":
        report = environment_report(_model_root())
        return report if args.as_json else diagnostics_text(report)
    if args.command == "backends":
        availability = transformers_availability()
        llama_availability = llama_cpp_availability()
        return {
            "backends": [
                {
                    "backend_id": availability.backend_id,
                    "available": availability.available,
                    "version": availability.detected_version,
                    "missing_packages": availability.missing_packages,
                    "reason": availability.reason,
                    "supported_devices": [
                        item.identifier for item in availability.supported_devices
                    ],
                },
                {
                    "backend_id": llama_availability.backend_id,
                    "available": llama_availability.available,
                    "version": llama_availability.detected_version,
                    "missing_packages": llama_availability.missing_packages,
                    "reason": llama_availability.reason,
                    "supported_devices": [
                        item.identifier for item in llama_availability.supported_devices
                    ],
                    "warnings": llama_availability.warnings,
                },
            ]
        }
    if args.command == "devices":
        return {
            "devices": [
                {
                    "type": item.device_type,
                    "index": item.index,
                    "name": item.display_name,
                    "memory": item.total_memory,
                    "recommended": item.recommended,
                    "warnings": item.warnings,
                }
                for item in detect_devices()
            ]
        }
    if args.command == "config":
        load_config(args.path)
        return {"status": "valid", "path": str(args.path) if args.path else "defaults"}
    if args.command == "models":
        result = discover_models(_model_root())
        if args.action == "validate":
            target = Path(args.target)
            if not target.exists():
                item = next(
                    (model for model in result.models if model.model_id == args.target), None
                )
                if item is None:
                    raise LinaCoreError(f"Model veya manifest bulunamadı: {args.target}")
                target = item.manifest_path
            validation = load_model_manifest(target, _model_root())
            return {
                "valid": validation.valid,
                "status": validation.status,
                "error": validation.error,
            }
        return {
            "models": [_model_dict(item) for item in result.models],
            "invalid_manifests": [str(path) for path in result.invalid_manifests],
            "duplicate_ids": result.duplicate_ids,
            "lazy": True,
        }
    if args.command == "runtime" and args.action == "start":
        return {
            "status": "not-implemented",
            "message": "Bu özellik henüz uygulanmadı.",
            "network_service_started": False,
        }
    if args.command == "runtime" and args.action == "load":
        model = _find_model(args.model_id)
        provider = provider_from_manifest(model.manifest_path, _model_root())
        provider.load()
        try:
            return {"status": "loaded-for-command", "health": provider.health().details}
        finally:
            provider.unload()
    if args.command == "runtime" and args.action == "unload":
        return {
            "status": "not-running",
            "message": "Kalıcı runtime servisi çalışmıyor; yüklü model yok.",
        }
    if args.command == "runtime":
        runtime = build_runtime(start=True)
        try:
            response = runtime.infer(InferenceRequest((ChatMessage(ChatRole.USER, "smoke"),)))
            return {
                "status": "ok",
                "provider": "mock",
                "response": response.text,
                "disclaimer": "Yalnızca mimari doğrulama; gerçek model değildir.",
            }
        finally:
            runtime.shutdown()
    if args.command == "inference":
        return _real_inference(args)
    if args.command == "benchmark":
        raise LinaCoreError("Benchmark için yüklenmiş gerçek bir model gereklidir.")
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
    raise LinaCoreError("Desteklenmeyen komut.")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        result = _execute(args)
        print(
            result
            if isinstance(result, str)
            else json.dumps(result, ensure_ascii=False, default=str)
        )
        return 0
    except (LinaCoreError, OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        if args.debug:
            raise
        print(
            json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False), file=sys.stderr
        )
        return 2
