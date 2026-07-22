import json
import re
from pathlib import Path

from lina_core.core.exceptions import DatasetError
from lina_core.datasets.contracts import DatasetManifest, DatasetRecord
from lina_core.security.limits import ResourceLimits
from lina_core.security.secrets import contains_secret

_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[A-Z]{2})?$")


def validate_record(record: DatasetRecord) -> None:
    if not record.record_id or not record.output.strip():
        raise DatasetError("Record id and output are required")
    if not record.source.strip():
        raise DatasetError("Record source is required")
    if not _LANGUAGE.fullmatch(record.language):
        raise DatasetError("Invalid language code")
    values = [record.output, record.instruction or "", record.input or ""]
    if sum(len(v.encode("utf-8")) for v in values) > ResourceLimits().max_dataset_record_size:
        raise DatasetError("Record is too large")
    if any(contains_secret(v) for v in values):
        raise DatasetError("Record may contain a secret")
    if record.metadata.get("private") is True:
        raise DatasetError("Private data is not accepted")


def validate_records(records: list[DatasetRecord]) -> None:
    seen: set[str] = set()
    for record in records:
        validate_record(record)
        if record.record_id in seen:
            raise DatasetError(f"Duplicate record: {record.record_id}")
        seen.add(record.record_id)


def load_manifest(path: Path) -> DatasetManifest:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetError(f"Cannot read manifest: {exc}") from exc
    try:
        manifest = DatasetManifest(**data)
    except TypeError as exc:
        raise DatasetError(f"Invalid manifest: {exc}") from exc
    validate_manifest(manifest)
    return manifest


def validate_manifest(item: DatasetManifest) -> None:
    if not item.dataset_id or item.record_count < 0 or not item.source or not item.license:
        raise DatasetError("Invalid dataset manifest")
    if not _LANGUAGE.fullmatch(item.language):
        raise DatasetError("Invalid language code")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", item.checksum):
        raise DatasetError("Checksum must be sha256")
