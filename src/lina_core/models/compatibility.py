import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import perf_counter
from typing import Any, TypeVar

from lina_core.backends.transformers.model import TransformersModelProvider
from lina_core.inference.request import ChatMessage, ChatRole, InferenceRequest
from lina_core.inference.sampling import GenerationOptions
from lina_core.inference.streaming import StreamEventType
from lina_core.security.limits import ResourceLimits
from lina_core.security.paths import safe_resolve

T = TypeVar("T")


class CompatibilityStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True, slots=True)
class CompatibilityCheck:
    name: str
    status: CompatibilityStatus
    duration_seconds: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ModelCompatibilityReport:
    schema_version: int
    model_id: str
    model_version: str
    source_revision: str | None
    license: str
    backend: str
    device: str | None
    dtype: str | None
    status: CompatibilityStatus
    started_at: datetime
    completed_at: datetime
    checks: tuple[CompatibilityCheck, ...]
    python_version: str = field(default_factory=platform.python_version)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str)


class TransformersCompatibilityRunner:
    def __init__(
        self,
        provider: TransformersModelProvider,
        *,
        model_version: str,
        source_revision: str | None,
        license_name: str,
    ) -> None:
        self.provider = provider
        self.model_version = model_version
        self.source_revision = source_revision
        self.license_name = license_name
        self._checks: list[CompatibilityCheck] = []

    def run(self, prompt: str = "The quick brown fox") -> ModelCompatibilityReport:
        started_at = datetime.now(UTC)
        overall_error: str | None = None
        generated_text = ""
        try:
            self._record("load", self._load)
            generated_text = self._record("generation", lambda: self._generation(prompt)) or ""
            self._record("streaming", lambda: self._streaming(prompt))
            self._record("cancellation", lambda: self._cancellation(prompt))
            self._record("stop_sequence", lambda: self._stop_sequence(prompt, generated_text))
            self._record("health", self._health)
            self._record("resources", self._resources)
        except Exception as exc:
            overall_error = self._safe_error(exc)
        finally:
            self._record("unload", self._unload)
        status = (
            CompatibilityStatus.PASSED
            if self._checks
            and all(check.status is CompatibilityStatus.PASSED for check in self._checks)
            else CompatibilityStatus.FAILED
        )
        health = self.provider.health().details
        return ModelCompatibilityReport(
            1,
            self.provider.metadata.model_id,
            self.model_version,
            self.source_revision,
            self.license_name,
            "transformers",
            health.get("device"),
            health.get("dtype"),
            status,
            started_at,
            datetime.now(UTC),
            tuple(self._checks),
            error=overall_error,
        )

    def _record(self, name: str, operation: Callable[[], T]) -> T | None:
        started = perf_counter()
        try:
            result = operation()
            details = result if isinstance(result, dict) else {}
            self._checks.append(
                CompatibilityCheck(
                    name,
                    CompatibilityStatus.PASSED,
                    perf_counter() - started,
                    details,
                )
            )
            return result
        except Exception as exc:
            self._checks.append(
                CompatibilityCheck(
                    name,
                    CompatibilityStatus.FAILED,
                    perf_counter() - started,
                    error=self._safe_error(exc),
                )
            )
            raise

    def _load(self) -> dict[str, Any]:
        self.provider.load()
        return {"loaded": self.provider.is_loaded}

    def _generation(self, prompt: str) -> str:
        response = self.provider.generate(self._request(prompt, "compat-generate"))
        if not response.text or response.usage.completion_tokens <= 0:
            raise RuntimeError("Model metin veya completion token üretmedi.")
        return response.text

    def _streaming(self, prompt: str) -> dict[str, Any]:
        events = list(
            self.provider.stream_generate(self._request(prompt, "compat-stream", stream=True))
        )
        if not events or events[0].type is not StreamEventType.STARTED:
            raise RuntimeError("Streaming started eventi üretmedi.")
        if events[-1].type is not StreamEventType.COMPLETED:
            raise RuntimeError("Streaming completed eventiyle bitmedi.")
        text = "".join(event.text for event in events if event.type is StreamEventType.TOKEN)
        return {"event_count": len(events), "characters": len(text)}

    def _cancellation(self, prompt: str) -> dict[str, Any]:
        request = self._request(prompt, "compat-cancel", stream=True, max_tokens=64)
        events = self.provider.stream_generate(request)
        observed: list[StreamEventType] = []
        for event in events:
            observed.append(event.type)
            if event.type is StreamEventType.STARTED:
                self.provider.cancel(request.request_id, "compatibility-test")
            if event.type is StreamEventType.CANCELLED:
                break
        if not observed or observed[-1] is not StreamEventType.CANCELLED:
            raise RuntimeError("Aktif stream cancellation ile sonlanmadı.")
        return {"terminal_event": observed[-1]}

    def _stop_sequence(self, prompt: str, generated_text: str) -> dict[str, Any]:
        stop = generated_text[:1]
        if not stop:
            raise RuntimeError("Stop sequence doğrulaması için çıktı üretilemedi.")
        response = self.provider.generate(self._request(prompt, "compat-stop", stop=(stop,)))
        if stop in response.text or response.finish_reason != "stop":
            raise RuntimeError("Stop sequence doğru uygulanmadı.")
        return {"finish_reason": response.finish_reason}

    def _health(self) -> dict[str, Any]:
        health = self.provider.health()
        if not health.details.get("loaded"):
            raise RuntimeError("Health modeli loaded olarak raporlamadı.")
        return {"status": health.status, "loaded": True}

    def _resources(self) -> dict[str, Any]:
        report = self.provider.resource_usage()
        if report.model_load_seconds is None or report.generated_tokens <= 0:
            raise RuntimeError("Resource report load süresi veya token sayısı içermiyor.")
        return report.to_dict()

    def _unload(self) -> dict[str, Any]:
        self.provider.unload()
        if self.provider.is_loaded:
            raise RuntimeError("Provider unload sonrasında loaded durumda kaldı.")
        return {"loaded": False}

    def _request(
        self,
        prompt: str,
        request_id: str,
        *,
        stream: bool = False,
        max_tokens: int = 12,
        stop: tuple[str, ...] = (),
    ) -> InferenceRequest:
        return InferenceRequest(
            (ChatMessage(ChatRole.USER, prompt),),
            GenerationOptions(
                max_new_tokens=max_tokens,
                temperature=0,
                seed=42,
                stop=stop,
            ),
            request_id=request_id,
            model_id=self.provider.metadata.model_id,
            stream=stream,
        )

    def _safe_error(self, exc: Exception) -> str:
        messages: list[str] = []
        current: BaseException | None = exc
        while current is not None and len(messages) < 4:
            message = str(current)
            if message and message not in messages:
                messages.append(message)
            current = current.__cause__ or current.__context__
        return " <- ".join(messages).replace(
            self.provider.config.model_path, Path(self.provider.config.model_path).name
        )


def write_compatibility_report(
    report: ModelCompatibilityReport,
    report_root: Path,
    filename: str,
) -> Path:
    target = safe_resolve(report_root, filename)
    payload = report.to_json() + "\n"
    if len(payload.encode("utf-8")) > ResourceLimits().max_report_size:
        raise ValueError("Compatibility report boyut sınırını aşıyor.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")
    return target
