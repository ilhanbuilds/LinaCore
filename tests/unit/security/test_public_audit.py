from pathlib import Path

from scripts.audit_public_repository import audit


def test_audit_clean_text(tmp_path: Path) -> None:
    path = tmp_path / "safe.md"
    path.write_text("Public documentation", encoding="utf-8")
    report = audit((path,))
    assert report.clean


def test_audit_detects_local_windows_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("C:/Users/example/private", encoding="utf-8")
    assert audit((path,)).findings[0].category == "local_path"


def test_audit_detects_local_unix_path(tmp_path: Path) -> None:
    path = tmp_path / "bad.md"
    path.write_text("/home/example/private", encoding="utf-8")
    assert audit((path,)).findings[0].category == "local_path"


def test_audit_detects_model_weight(tmp_path: Path) -> None:
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF")
    assert audit((path,)).findings[0].category == "forbidden_artifact"


def test_audit_detects_private_key(tmp_path: Path) -> None:
    path = tmp_path / "secret.txt"
    path.write_text("-----BEGIN PRIVATE KEY-----", encoding="utf-8")
    assert audit((path,)).findings[0].category == "private_key"


def test_audit_detects_github_token(tmp_path: Path) -> None:
    path = tmp_path / "secret.txt"
    path.write_text("ghp_abcdefghijklmnopqrstuvwxyz1234", encoding="utf-8")
    assert audit((path,)).findings[0].category == "github_token"


def test_audit_detects_large_file(tmp_path: Path) -> None:
    path = tmp_path / "large.txt"
    path.write_text("x" * 1_000_001, encoding="utf-8")
    assert audit((path,)).findings[0].category == "large_file"


def test_audit_report_counts_candidates(tmp_path: Path) -> None:
    paths = tuple(tmp_path / f"{index}.md" for index in range(3))
    for path in paths:
        path.write_text("safe", encoding="utf-8")
    assert audit(paths).files_checked == 3
