import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

from lina_core.core.exceptions import TrainingError


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    model_id: str
    revision: str
    license: str
    parameter_count: int
    multilingual_rating: int
    context_length: int
    architecture: str
    chat_template: bool
    transformers_compatible: bool
    lora_compatible: bool
    gguf_convertible: bool
    estimated_download_gb: float
    estimated_inference_ram_gb: float
    cpu_inference: str
    source_url: str
    license_url: str
    limitation: str


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    candidate: ModelCandidate
    score: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelSelectionReport:
    schema_version: int
    recommended_model_id: str
    ranked: tuple[RankedCandidate, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def load_model_candidates(path: Path) -> list[ModelCandidate]:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        raw_candidates = data["candidate"]
        if not isinstance(raw_candidates, list):
            raise TrainingError("Model aday listesi geçersiz.")
        candidates = [ModelCandidate(**item) for item in raw_candidates]
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as exc:
        raise TrainingError(f"Model aday raporu okunamadı: {exc}") from exc
    if len(candidates) < 2 or len({item.model_id for item in candidates}) != len(candidates):
        raise TrainingError("Model aday raporu en az iki benzersiz model içermeli.")
    return candidates


def rank_model_candidates(candidates: list[ModelCandidate]) -> ModelSelectionReport:
    ranked = sorted((_score(candidate) for candidate in candidates), key=_rank_key)
    return ModelSelectionReport(1, ranked[0].candidate.model_id, tuple(ranked))


def _score(candidate: ModelCandidate) -> RankedCandidate:
    score = 0
    reasons: list[str] = []
    if candidate.license == "Apache-2.0":
        score += 2
        reasons.append("public kullanıma uygun Apache-2.0 lisans")
    if 300_000_000 <= candidate.parameter_count <= 600_000_000:
        score += 2
        reasons.append("300M-600M hedef aralığı")
    score += candidate.multilingual_rating
    reasons.append(f"çok dilli kapasite puanı {candidate.multilingual_rating}/3")
    for supported, points, label in (
        (candidate.chat_template, 1, "chat template"),
        (candidate.transformers_compatible, 1, "Transformers"),
        (candidate.lora_compatible, 1, "LoRA"),
        (candidate.gguf_convertible, 1, "GGUF dönüşüm"),
        (candidate.estimated_download_gb < 2, 1, "2 GB altı indirme"),
    ):
        if supported:
            score += points
            reasons.append(label)
    return RankedCandidate(candidate, score, tuple(reasons))


def _rank_key(item: RankedCandidate) -> tuple[int, float, int, str]:
    return (
        -item.score,
        item.candidate.estimated_download_gb,
        item.candidate.parameter_count,
        item.candidate.model_id,
    )
