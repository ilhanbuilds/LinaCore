import time
from uuid import uuid4

from lina_core.evaluation.metrics import (
    contains_expected,
    exact_match,
    normalized_exact_match,
    response_length,
)
from lina_core.evaluation.models import EvaluationResult, EvaluationSample, MetricResult
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.runtime.service import RuntimeService


class EvaluationRunner:
    def __init__(self, runtime: RuntimeService) -> None:
        self.runtime = runtime

    def run(self, samples: list[EvaluationSample], dataset_id: str = "adhoc") -> EvaluationResult:
        started = time.perf_counter()
        metrics: list[MetricResult] = []
        failures = 0
        outputs: dict[str, str] = {}
        for sample in samples:
            messages = sample.messages or (ChatMessage(ChatRole.USER, sample.prompt or ""),)
            request = InferenceRequest(messages)
            first = self.runtime.infer(request).text
            second = self.runtime.infer(request).text
            outputs[sample.sample_id] = first
            metrics.append(response_length(first))
            metrics.append(
                MetricResult(
                    "deterministic_repeatability", float(first == second), passed=first == second
                )
            )
            if sample.expected_output is not None:
                metrics.extend(
                    (
                        exact_match(first, sample.expected_output),
                        normalized_exact_match(first, sample.expected_output),
                        contains_expected(first, sample.expected_output),
                    )
                )
                failures += int(sample.expected_output not in first)
        return EvaluationResult(
            str(uuid4()),
            self.runtime.active_model_id or "none",
            dataset_id,
            tuple(metrics),
            time.perf_counter() - started,
            len(samples),
            failures,
        )
