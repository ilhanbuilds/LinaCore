import json
from pathlib import Path

import pytest

from lina_core.cli import main


@pytest.mark.parametrize(
    "args",
    [["version"], ["health"], ["models", "list"], ["runtime", "smoke"], ["config", "validate"]],
)
def test_cli_success(args: list[str], capsys: pytest.CaptureFixture[str]):
    assert main(args) == 0 and json.loads(capsys.readouterr().out)


def test_cli_config_file(capsys: pytest.CaptureFixture[str]):
    assert main(["config", "validate", "configs/development.toml"]) == 0


def test_cli_runtime_start(capsys: pytest.CaptureFixture[str]):
    assert main(["runtime", "start"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "not-implemented"


def test_cli_dataset(manifest_file: Path, capsys: pytest.CaptureFixture[str]):
    assert main(["dataset", "validate", str(manifest_file)]) == 0


def test_cli_dataset_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    assert main(["dataset", "validate", str(bad)]) == 2


def test_cli_evaluation(capsys: pytest.CaptureFixture[str]):
    assert main(["evaluation", "run", "configs/evaluation.example.toml"]) == 0
