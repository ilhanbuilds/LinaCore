"""Compare Qwen base, Lina v0.1, and optionally Lina v0.2 on the same holdout."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from lina_core.backends.device import DTypePolicy
from lina_core.backends.transformers.config import TransformersBackendConfig
from lina_core.backends.transformers.model import TransformersModelProvider
from lina_core.evaluation.behavior import (
    compare_behavior_models,
    load_behavior_samples,
    write_behavior_report,
)
from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse, TokenUsage
from lina_core.models.base import ModelProvider
from lina_core.models.local import provider_from_manifest
from lina_core.models.metadata import ModelMetadata
from lina_core.models.mock import MockModelProvider


class RecordedProvider(MockModelProvider):
    def __init__(self, model_id: str, outputs: dict[str, dict[str, Any]]) -> None:
        super().__init__()
        self.model_id = model_id
        self.outputs = outputs

    @property
    def metadata(self) -> ModelMetadata:
        return replace(super().metadata, model_id=self.model_id, display_name=self.model_id)

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        if not self.is_loaded:
            return super().generate(request)
        prompt = request.messages[-1].content
        output = self.outputs[prompt]
        return InferenceResponse(
            request.request_id,
            self.model_id,
            str(output["text"]),
            "stop",
            TokenUsage(int(output["prompt_tokens"]), int(output["completion_tokens"])),
            float(output["latency_seconds"]),
        )


def _base_provider(path: Path) -> TransformersModelProvider:
    return TransformersModelProvider(
        TransformersBackendConfig(
            model_path=str(path.resolve()),
            model_id="Qwen/Qwen2.5-0.5B-Instruct",
            display_name="Qwen2.5-0.5B-Instruct",
            revision="7ae557604adf67be50417f59c2c2f167def9a775",
            device="cpu",
            dtype=DTypePolicy.FLOAT32,
            local_files_only=True,
            allow_remote_downloads=False,
            trust_remote_code=False,
            chat_template_policy="required",
        ),
        context_length=32768,
        architecture="Qwen2ForCausalLM",
        source="Qwen/Qwen2.5-0.5B-Instruct",
        license_name="Apache-2.0",
    )


def _recorded_providers(path: Path) -> list[RecordedProvider]:
    data = json.loads(path.read_text(encoding="utf-8"))
    model_ids = [summary["model_id"] for summary in data["summaries"]]
    providers: list[RecordedProvider] = []
    for index, model_id in enumerate(model_ids):
        outputs = {item["prompt"]: item["outputs"][index] for item in data["items"]}
        providers.append(RecordedProvider(str(model_id), outputs))
    return providers


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-path", type=Path, default=Path("models/qwen2.5-0.5b-instruct"))
    parser.add_argument(
        "--v0-1-manifest",
        type=Path,
        default=Path("models/lina-135m-sft-v0.1/linacore-model.toml"),
    )
    parser.add_argument("--v0-2-manifest", type=Path)
    parser.add_argument(
        "--baseline-report",
        type=Path,
        help="Base ve v0.1 çıktılarını yeniden inference yapmadan kullanır.",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        default=Path("data/examples/lina_behavior_holdout_v0.2.jsonl"),
    )
    parser.add_argument("--report-root", type=Path, default=Path("reports"))
    parser.add_argument("--report", default="base-vs-v0.1-vs-v0.2.json")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    model_root = Path("models").resolve()
    if args.baseline_report is not None:
        providers: list[ModelProvider] = list(_recorded_providers(args.baseline_report))
    else:
        v0_1_provider = provider_from_manifest(args.v0_1_manifest.resolve(), model_root)
        # The v0.1 artifact accidentally recorded training sequence length as model context.
        v0_1_provider.context_length = 8192
        providers = [_base_provider(args.base_path), v0_1_provider]
    if args.v0_2_manifest is not None:
        providers.append(provider_from_manifest(args.v0_2_manifest.resolve(), model_root))
    report = compare_behavior_models(
        tuple(providers),
        load_behavior_samples(args.holdout),
        max_new_tokens=args.max_new_tokens,
        seed=42,
    )
    target = write_behavior_report(report, args.report_root.resolve(), args.report)
    print(
        json.dumps(
            {
                "report": str(target),
                "summaries": [asdict(summary) for summary in report.summaries],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
