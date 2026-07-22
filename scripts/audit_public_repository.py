"""Audit committable repository content for common public-release hazards."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 1_000_000
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".gguf",
    ".key",
    ".onnx",
    ".p12",
    ".pem",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
}
TEXT_SUFFIXES = {"", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
PATTERN_EXEMPT_PATHS = {
    "scripts/audit_public_repository.py",
    "src/lina_core/security/secrets.py",
    "src/lina_core/datasets/sft.py",
    "tests/unit/core/test_core.py",
    "tests/unit/security/test_public_audit.py",
    "tests/unit/datasets/test_sft_dataset.py",
}
PATTERNS = {
    "local_path": re.compile(r"(?i)(?:[A-Z]:[\\/](?:Users|Documents|Desktop)[\\/]|/home/|/Users/)"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "bearer_token": re.compile(r"(?i)bearer\s+[A-Za-z0-9._-]{20,}"),
}


@dataclass(frozen=True, slots=True)
class AuditFinding:
    path: str
    category: str
    detail: str


@dataclass(frozen=True, slots=True)
class AuditReport:
    files_checked: int
    findings: tuple[AuditFinding, ...]

    @property
    def clean(self) -> bool:
        return not self.findings


def repository_files() -> tuple[Path, ...]:
    output = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={ROOT.as_posix()}",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return tuple(ROOT / item.decode("utf-8") for item in output.split(b"\0") if item)


def audit(paths: tuple[Path, ...] | None = None) -> AuditReport:
    candidates = paths if paths is not None else repository_files()
    findings: list[AuditFinding] = []
    for path in candidates:
        try:
            relative = path.relative_to(ROOT).as_posix()
        except ValueError:
            relative = path.name
        if path.suffix.casefold() in FORBIDDEN_SUFFIXES:
            findings.append(AuditFinding(relative, "forbidden_artifact", path.suffix))
            continue
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            findings.append(AuditFinding(relative, "large_file", str(size)))
        if path.suffix.casefold() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            findings.append(AuditFinding(relative, "invalid_utf8", "text file is not UTF-8"))
            continue
        if relative in PATTERN_EXEMPT_PATHS:
            continue
        for category, pattern in PATTERNS.items():
            if pattern.search(content):
                findings.append(AuditFinding(relative, category, "pattern matched"))
    return AuditReport(len(candidates), tuple(findings))


def main() -> int:
    report = audit()
    print(json.dumps({**asdict(report), "clean": report.clean}, ensure_ascii=False, indent=2))
    return 0 if report.clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
