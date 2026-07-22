from lina_core.core.exceptions import LinaCoreError
from lina_core.inference.request import InferenceRequest
from lina_core.runtime.service import RuntimeService
from lina_core.serving.contracts import ApiResponse


class LocalService:
    def __init__(self, runtime: RuntimeService) -> None:
        self.runtime = runtime

    def health(self) -> ApiResponse:
        status = self.runtime.health()
        return ApiResponse(True, {"status": status.status, "message": status.message})

    def models(self) -> ApiResponse:
        return ApiResponse(
            True, {"models": self.runtime.models.list(), "active": self.runtime.active_model_id}
        )

    def infer(self, request: InferenceRequest) -> ApiResponse:
        try:
            response = self.runtime.infer(request)
            return ApiResponse(
                True,
                {
                    "request_id": response.request_id,
                    "model_id": response.model_id,
                    "text": response.text,
                    "finish_reason": response.finish_reason,
                },
            )
        except LinaCoreError as exc:
            return ApiResponse(False, error=str(exc))

    def shutdown(self) -> ApiResponse:
        self.runtime.shutdown()
        return ApiResponse(True, {"status": "stopped"})
