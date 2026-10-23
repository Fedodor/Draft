#!/usr/bin/env python
"""Deterministic working-tree secret scan.

Scans every tracked file, plus untracked files that are not git-ignored, for
credential-shaped strings. Exits non-zero when a finding is not allowlisted.

Usage:
    python scripts/secret_scan.py [--root .] [--all] [--json]

Notes:
    * Only the Python standard library is used, so this runs anywhere.
    * Findings are matched by path and by a small placeholder filter, so real
      keys fail while examples and documentation do not.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

MAX_FILE_BYTES = 2 * 1024 * 1024

PATTERNS = [
    (
        "private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
    ),
    ("aws access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "github token",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b"),
    ),
    ("github pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("google api key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("stripe secret key", re.compile(r"\bsk_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    (
        "jwt",
        re.compile(
            r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"
        ),
    ),
    (
        "credential assignment",
        re.compile(
            r"(?i)\b(?:api[_-]?key|apikey|secret|client[_-]?secret|access[_-]?token"
            r"|auth[_-]?token|password|passwd|stratz[_-]?key|opendota[_-]?key)\b"
            r"\s*[:=]\s*['\"]([^'\"\n]{12,})['\"]"
        ),
    ),
    (
        "credential query parameter",
        re.compile(r"(?i)[?&](?:api_?key|token|key)=([A-Za-z0-9_\-.]{16,})"),
    ),
]

PLACEHOLDER_HINTS = (
    "your_",
    "your-",
    "example",
    "placeholder",
    "changeme",
    "change-me",
    "change_me",
    "replace-with",
    "replace_with",
    "redacted",
    "dummy",
    "sample",
    "xxxx",
    "todo",
    "fake",
    "not-a-real",
)

ALLOWLISTED_PATHS = (
    ".env.example",
    "scripts/secret_scan.py",
    "ml/tests/test_no_hardcoded_keys.py",
    "ml/tests/test_settings.py",
    "docs/",
    "README.md",
)

TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".env",
    ".example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonc",
    ".jsx",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".rs",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    kind: str
    preview: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: [{self.kind}] {self.preview}"


def _run_git(root: Path, args: Sequence[str]) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _git_files(root: Path) -> Optional[List[Path]]:
    tracked = _run_git(root, ["ls-files", "-z"])
    if tracked is None:
        return None
    others = _run_git(root, ["ls-files", "--others", "--exclude-standard", "-z"]) or ""
    names = [name for name in (tracked + others).split("\0") if name]
    return [root / name for name in names]


def _walk_files(root: Path) -> List[Path]:
    ignored_dirs = {
        ".git",
        "node_modules",
        "target",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "dist",
    }
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_dirs for part in path.relative_to(root).parts):
            continue
        files.append(path)
    return files


def _is_allowlisted(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    return any(token in normalized for token in ALLOWLISTED_PATHS)


def _looks_textual(path: Path, data: bytes) -> bool:
    if b"\0" in data[:4096]:
        return False
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return True


def _placeholder(value: str) -> bool:
    lowered = value.strip().strip("'\"").lower()
    if lowered.startswith("<") and lowered.endswith(">"):
        return True
    return any(hint in lowered for hint in PLACEHOLDER_HINTS)


def _scan_text(relative: str, text: str) -> List[Finding]:
    findings: List[Finding] = []
    lines = text.splitlines()
    for kind, pattern in PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(1) if match.groups() else match.group(0)
            if _placeholder(value):
                continue
            line = text.count("\n", 0, match.start()) + 1
            preview = lines[line - 1].strip() if line - 1 < len(lines) else ""
            findings.append(
                Finding(
                    path=relative.replace("\\", "/"),
                    line=line,
                    kind=kind,
                    preview=preview[:120],
                )
            )
    return findings


def scan(root: Path, use_all: bool = False) -> List[Finding]:
    root = root.resolve()
    if use_all:
        files = _walk_files(root)
    else:
        files = _git_files(root)
        if files is None:
            files = _walk_files(root)

    findings: List[Finding] = []
    for path in files:
        try:
            relative = str(path.relative_to(root))
        except ValueError:
            relative = str(path)
        if _is_allowlisted(relative):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size == 0 or size > MAX_FILE_BYTES:
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if not _looks_textual(path, data):
            continue
        text = data.decode("utf-8", errors="replace")
        findings.extend(_scan_text(relative, text))
    return findings


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Scan the tree for credentials.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Walk the whole tree instead of only git-visible files.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    findings = scan(Path(args.root), use_all=args.all)
    if args.json:
        print(json.dumps([f.__dict__ for f in findings], indent=2))
    elif findings:
        for finding in findings:
            print(finding.render())
    else:
        print("secret scan: clean")

    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
