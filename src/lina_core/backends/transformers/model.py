import gc
import time
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import Any

from lina_core.backends.device import select_device
from lina_core.backends.errors import BackendUnavailableError, GenerationError
from lina_core.backends.transformers.availability import transformers_availability
from lina_core.backends.transformers.config import TransformersBackendConfig
from lina_core.backends.transformers.device import torch_dtype
from lina_core.backends.transformers.generation import (
    CancellationStoppingCriteria,
    apply_seed,
    generation_kwargs,
)
from lina_core.backends.transformers.loader import load_modules
from lina_core.core.exceptions import ModelLoadError
from lina_core.diagnostics.hardware import detect_devices
from lina_core.inference.chat_template import ChatTemplateRenderer
from lina_core.inference.prompt import prepare_prompt
from lina_core.inference.request import InferenceRequest
from lina_core.inference.response import InferenceResponse, TokenUsage
from lina_core.inference.stop import apply_stop_sequences
from lina_core.inference.streaming import StreamEvent, StreamEventType
from lina_core.models.base import ModelProvider
from lina_core.models.metadata import ModelCapability, ModelMetadata
from lina_core.runtime.cancellation import CancellationRegistry
from lina_core.runtime.health import HealthLevel, HealthStatus
from lina_core.runtime.resources import ResourceReport, process_memory_bytes
from lina_core.tokenizer.base import TokenizerProvider

from .tokenizer import TransformersTokenizerProvider


class TransformersModelProvider(ModelProvider):
    def __init__(
        self,
        config: TransformersBackendConfig,
        *,
        context_length: int = 2048,
        architecture: str = "causal-lm",
        source: str = "local",
        license_name: str = "unknown",
        capabilities: frozenset[ModelCapability] | None = None,
        model: Any | None = None,
        tokenizer: TransformersTokenizerProvider | None = None,
        torch_module: Any | None = None,
    ) -> None:
        self.config = config
        self.context_length = context_length
        self.architecture = architecture
        self.source = source
        self.license_name = license_name
        self.capabilities = capabilities or frozenset(
            {
                ModelCapability.TEXT_GENERATION,
                ModelCapability.CHAT,
                ModelCapability.STREAMING,
            }
        )
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch_module
        self._selection: Any | None = None
        self._cancellations = CancellationRegistry()
        self._load_seconds: float | None = None
        self._generated_tokens = 0
        self._last_latency: float | None = None
        self._last_error: str | None = None

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            self.config.model_id,
            self.config.display_name,
            "local",
            self.architecture,
            self.context_length,
            self.capabilities,
            self.source,
            self.license_name,
            loaded=self.is_loaded,
            quantization=self.config.quantization.value,
        )

    def load(self) -> None:
        if self.is_loaded:
            return
        started = time.perf_counter()
        availability = transformers_availability()
        if not availability.available:
            raise BackendUnavailableError(availability.reason)
        if self.config.local_files_only and not Path(self.config.model_path).exists():
            raise ModelLoadError("Yerel model yolu bulunamadı.")
        transformers, torch = load_modules()
        self._torch = torch
        self._selection = select_device(
            self.config.device,
            detect_devices(torch),
            self.config.dtype,
            self.config.quantization,
        )
        try:
            tokenizer = TransformersTokenizerProvider(self.config)
            tokenizer.load()
            self._tokenizer = tokenizer
            dtype_keyword = (
                "dtype"
                if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5
                else "torch_dtype"
            )
            self._model = transformers.AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                revision=self.config.revision,
                local_files_only=self.config.local_files_only,
                trust_remote_code=False,
                low_cpu_mem_usage=self.config.low_cpu_mem_usage,
                use_safetensors=self.config.use_safetensors,
                **{dtype_keyword: torch_dtype(self._selection, torch)},
            )
            self._model.to(self._selection.device.identifier)
            self._model.eval()
            self._load_seconds = time.perf_counter() - started
        except (OSError, ValueError, RuntimeError) as exc:
            self._last_error = str(exc)
            self.unload()
            raise ModelLoadError(f"Transformers modeli yüklenemedi: {exc}") from exc

    def unload(self) -> None:
        self._cancellations.cancel_all("unload")
        self._model = None
        if self._tokenizer is not None:
            self._tokenizer.unload()
        self._tokenizer = None
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def create_tokenizer(self) -> TokenizerProvider:
        if self._tokenizer is None:
            raise ModelLoadError("Tokenizer yüklenmedi.")
        return self._tokenizer

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        model, tokenizer, torch, selection = self._require_loaded()
        token = self._cancellations.create(request.request_id)
        if token.cancelled:
            raise GenerationError("İstek inference başlamadan iptal edildi.")
        started = time.perf_counter()
        try:
            transformers, _ = load_modules()
            prepared = self._prepare(request, tokenizer)
            inputs = tokenizer.raw_tokenizer(prepared.rendered.text, return_tensors="pt")
            inputs = {key: value.to(selection.device.identifier) for key, value in inputs.items()}
            apply_seed(request.options.seed, torch)
            generate_options = generation_kwargs(request.options)
            generate_options["stopping_criteria"] = transformers.StoppingCriteriaList(
                [CancellationStoppingCriteria(token)]
            )
            with torch.inference_mode():
                output = model.generate(**inputs, **generate_options)
            prompt_length = int(inputs["input_ids"].shape[-1])
            generated = output[0][prompt_length:].tolist()
            text = tokenizer.decode(generated)
            stopped = apply_stop_sequences(text, request.options.stop)
            completion_tokens = tokenizer.count_tokens(stopped.text)
            latency = time.perf_counter() - started
            self._generated_tokens += completion_tokens
            self._last_latency = latency
            if token.cancelled:
                finish = "cancelled"
            elif stopped.stopped or completion_tokens < request.options.max_new_tokens:
                finish = "stop"
            else:
                finish = "length"
            return InferenceResponse(
                request.request_id,
                self.config.model_id,
                stopped.text,
                finish,
                TokenUsage(prepared.budget.prompt_tokens, completion_tokens),
                latency,
            )
        except (ModelLoadError, GenerationError):
            raise
        except (RuntimeError, TypeError, ValueError) as exc:
            self._last_error = str(exc)
            raise GenerationError(f"Generation başarısız oldu: {exc}") from exc
        finally:
            self._cancellations.finish(request.request_id)

    def stream_generate(self, request: InferenceRequest) -> Iterator[StreamEvent]:
        model, tokenizer, torch, selection = self._require_loaded()
        transformers, _ = load_modules()
        cancellation = self._cancellations.create(request.request_id)
        prepared = self._prepare(request, tokenizer)
        inputs = tokenizer.raw_tokenizer(prepared.rendered.text, return_tensors="pt")
        inputs = {key: value.to(selection.device.identifier) for key, value in inputs.items()}
        streamer = transformers.TextIteratorStreamer(
            tokenizer.raw_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kwargs = {**inputs, **generation_kwargs(request.options), "streamer": streamer}
        kwargs["stopping_criteria"] = transformers.StoppingCriteriaList(
            [CancellationStoppingCriteria(cancellation)]
        )
        failure: list[Exception] = []

        def worker() -> None:
            try:
                apply_seed(request.options.seed, torch)
                with torch.inference_mode():
                    model.generate(**kwargs)
            except Exception as exc:  # Backend worker boundary.
                failure.append(exc)
                streamer.end()

        thread = Thread(target=worker, name=f"lina-stream-{request.request_id}", daemon=True)
        thread.start()
        index = 0
        full_text = ""
        yield StreamEvent(StreamEventType.STARTED, request.request_id, index=index)
        try:
            for piece in streamer:
                if cancellation.cancelled:
                    index += 1
                    yield StreamEvent(StreamEventType.CANCELLED, request.request_id, index=index)
                    return
                full_text += str(piece)
                index += 1
                yield StreamEvent(StreamEventType.TOKEN, request.request_id, str(piece), index)
            index += 1
            if failure:
                self._last_error = str(failure[0])
                yield StreamEvent(
                    StreamEventType.ERROR, request.request_id, "Backend streaming hatası.", index
                )
            else:
                self._generated_tokens += tokenizer.count_tokens(full_text)
                yield StreamEvent(StreamEventType.COMPLETED, request.request_id, index=index)
        finally:
            thread.join(5.0)
            self._cancellations.finish(request.request_id)

    def cancel(self, request_id: str, reason: str = "user") -> bool:
        return self._cancellations.cancel(request_id, reason)

    def health(self) -> HealthStatus:
        availability = transformers_availability()
        level = HealthLevel.HEALTHY if self.is_loaded else HealthLevel.DEGRADED
        details = {
            "backend": "transformers",
            "backend_available": availability.available,
            "device": self._selection.device.identifier if self._selection else None,
            "dtype": self._selection.dtype.value if self._selection else None,
            "loaded": self.is_loaded,
            "current_requests": self._cancellations.active_count,
            "model_path": Path(self.config.model_path).name,
            "last_error": self._last_error,
        }
        return HealthStatus(
            level, "Model yüklü." if self.is_loaded else "Model yüklü değil.", details
        )

    def resource_usage(self) -> ResourceReport:
        return ResourceReport(
            process_memory_bytes(),
            self._load_seconds,
            active_requests=self._cancellations.active_count,
            generated_tokens=self._generated_tokens,
            last_inference_latency=self._last_latency,
        )

    def _require_loaded(self) -> tuple[Any, TransformersTokenizerProvider, Any, Any]:
        if (
            not self.is_loaded
            or self._tokenizer is None
            or self._torch is None
            or self._selection is None
        ):
            raise ModelLoadError("Transformers modeli yüklenmedi.")
        return self._model, self._tokenizer, self._torch, self._selection

    def _prepare(self, request: InferenceRequest, tokenizer: TransformersTokenizerProvider) -> Any:
        renderer = ChatTemplateRenderer(
            tokenizer.raw_tokenizer,
            fallback_enabled=self.config.chat_template_policy == "fallback",
        )
        return prepare_prompt(
            request.messages,
            tokenizer,
            renderer,
            context_length=self.context_length,
            requested_output_tokens=request.options.max_new_tokens,
        )
