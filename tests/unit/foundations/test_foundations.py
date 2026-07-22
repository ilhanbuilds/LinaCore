from pathlib import Path

import pytest

from lina_core.core.exceptions import DatasetError, TrainingError
from lina_core.datasets.contracts import DatasetManifest, DatasetRecord
from lina_core.datasets.preprocessing import (
    DatasetSplit,
    fingerprint,
    normalize_text,
    quality_score,
)
from lina_core.datasets.validation import (
    load_manifest,
    validate_manifest,
    validate_record,
    validate_records,
)
from lina_core.evaluation.metrics import (
    contains_expected,
    exact_match,
    normalized_exact_match,
    response_length,
)
from lina_core.evaluation.models import EvaluationSample
from lina_core.evaluation.reports import write_report
from lina_core.evaluation.runner import EvaluationRunner
from lina_core.runtime.service import RuntimeService
from lina_core.training.checkpoints import CheckpointMetadata
from lina_core.training.config import TrainingConfig, load_training_config
from lina_core.training.runner import TrainingRunner


def record(**kwargs: object) -> DatasetRecord:
    return DatasetRecord(
        **{"record_id": "1", "output": "answer", "source": "test", "language": "en", **kwargs}
    )  # type: ignore[arg-type]


def test_record_valid():
    validate_record(record())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output": " "},
        {"source": ""},
        {"language": "english"},
        {"metadata": {"private": True}},
        {"output": "password=hunter2"},
    ],
)
def test_record_invalid(kwargs: dict[str, object]):
    with pytest.raises(DatasetError):
        validate_record(record(**kwargs))


def test_duplicates():
    with pytest.raises(DatasetError):
        validate_records([record(), record()])


def test_normalize():
    assert normalize_text(" e\u0301   x\r\n") == "é x"


def test_fingerprint_stable():
    assert fingerprint(record(output="X")) == fingerprint(record(output=" x "))


def test_quality():
    assert 0 < quality_score(record()) <= 1


def test_split():
    assert DatasetSplit().train == 0.8


def test_bad_split():
    with pytest.raises(ValueError):
        DatasetSplit(0.5, 0.5, 0.5)


def test_manifest(manifest_file: Path):
    assert load_manifest(manifest_file).dataset_id == "demo"


def test_bad_manifest_checksum():
    with pytest.raises(DatasetError):
        validate_manifest(DatasetManifest("x", "1", "train", 1, "en", "MIT", "bad", "src"))


@pytest.mark.parametrize("actual,expected,value", [("a", "a", 1), ("A", "a", 0)])
def test_exact(actual: str, expected: str, value: int):
    assert exact_match(actual, expected).value == value


def test_normalized_exact():
    assert normalized_exact_match(" A  B ", "a b").passed


def test_contains():
    assert contains_expected("hello world", "WORLD").passed


def test_response_length():
    assert response_length("abc").value == 3


def test_evaluation_sample_invalid():
    with pytest.raises(ValueError):
        EvaluationSample("x", "c", "en")


def test_evaluation_runner(runtime: RuntimeService):
    result = EvaluationRunner(runtime).run(
        [EvaluationSample("1", "c", "en", prompt="hello", expected_output="[mock] hello")]
    )
    assert result.failures == 0 and result.sample_count == 1


def test_report(runtime: RuntimeService, tmp_path: Path):
    result = EvaluationRunner(runtime).run([])
    assert write_report(result, tmp_path, "a.json").exists()


def training_config(**kwargs: object) -> TrainingConfig:
    return TrainingConfig(
        **{
            "run_name": "r",
            "model_id": "m",
            "tokenizer_id": "t",
            "dataset_manifest": "manifest.json",
            "output_dir": "out",
            **kwargs,
        }
    )  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epochs": 0},
        {"batch_size": 0},
        {"learning_rate": 0},
        {"warmup_ratio": 2},
        {"precision": "int4"},
    ],
)
def test_training_invalid(kwargs: dict[str, object]):
    with pytest.raises(TrainingError):
        training_config(**kwargs)


def test_training_load(tmp_path: Path):
    p = tmp_path / "t.toml"
    p.write_text(
        '[training]\nrun_name="r"\nmodel_id="m"\ntokenizer_id="t"\ndataset_manifest="d"\noutput_dir="o"'
    )
    assert load_training_config(p).run_name == "r"


def test_training_dry_run(tmp_path: Path, manifest_file: Path):
    result = TrainingRunner(tmp_path).dry_run(training_config(dataset_manifest="manifest.json"))
    assert result["training_started"] is False


def test_checkpoint(tmp_path: Path):
    assert CheckpointMetadata("c", "r", 0, 0, tmp_path, "sha256:x").resumable is False


def test_checkpoint_negative(tmp_path: Path):
    with pytest.raises(ValueError):
        CheckpointMetadata("c", "r", -1, 0, tmp_path, "x")
