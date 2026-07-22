from pathlib import Path

import pytest

from lina_core.models.mock import MockModelProvider
from lina_core.runtime.service import RuntimeService
from lina_core.tokenizer.mock import MockTokenizerProvider


@pytest.fixture
def runtime() -> RuntimeService:
    service = RuntimeService()
    service.register_model(MockModelProvider())
    service.register_tokenizer(MockTokenizerProvider())
    service.startup("mock", "mock")
    yield service
    if service.active_model_id:
        service.shutdown()


@pytest.fixture
def manifest_file(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"dataset_id":"demo","version":"1","split":"train","record_count":0,"language":"tr","license":"MIT","checksum":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","source":"test"}',
        encoding="utf-8",
    )
    return path
