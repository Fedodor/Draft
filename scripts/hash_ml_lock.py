#!/usr/bin/env python3
"""Write the hash-pinned ML lock from a pip freeze and downloaded wheels."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from packaging.utils import canonicalize_name, parse_wheel_filename

HEADER = [
    "# Reproducible ML environment for Python 3.12.13 (Windows x64).",
    "# Direct dependencies live in ml/requirements.txt; this file pins the",
    "# verified transitive set with SHA-256 hashes for the Windows x64 wheels.",
    "# Regenerate from a clean 3.12.13 venv with:",
    "#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/update_ml_lock.ps1 -Update",
    "# The script installs the current lock, runs ml/tests, then re-derives the",
    "# versions and hashes. Install with:",
    "#   python -m pip install --require-hashes -r requirements.lock",
]


def fail(message: str) -> None:
    raise SystemExit(f"error: {message}")


def read_pins(path: Path) -> list[tuple[str, str]]:
    pins: list[tuple[str, str]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            fail(f"{path}:{line_number}: expected a pinned requirement, got {line!r}")
        name, version = line.split("==", 1)
        if not name or not version or any(char.isspace() for char in version):
            fail(f"{path}:{line_number}: invalid pinned requirement {line!r}")
        pins.append((name, version))
    if not pins:
        fail(f"{path}: no pinned requirements found")
    return pins


def read_hashes(wheelhouse: Path) -> dict[tuple[str, str], str]:
    hashes: dict[tuple[str, str], str] = {}
    for wheel in sorted(wheelhouse.glob("*.whl")):
        try:
            name, version, _, _ = parse_wheel_filename(wheel.name)
        except Exception as exc:
            fail(f"{wheel.name}: not a valid wheel: {exc}")
        key = (canonicalize_name(name), str(version))
        if key in hashes:
            fail(f"multiple wheels for {name}=={version}: {wheel.name}")
        hashes[key] = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if not hashes:
        fail(f"{wheelhouse}: no wheels found")
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirements", type=Path, required=True, help="pip freeze output")
    parser.add_argument("--wheelhouse", type=Path, required=True, help="directory of downloaded wheels")
    parser.add_argument("--output", type=Path, required=True, help="hashed lock to write")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pins = read_pins(args.requirements)
    hashes = read_hashes(args.wheelhouse)

    expected = {(canonicalize_name(name), version) for name, version in pins}
    missing = expected - set(hashes)
    extra = set(hashes) - expected
    if missing:
        fail(f"no wheel hash for: {', '.join(f'{name}=={version}' for name, version in sorted(missing))}")
    if extra:
        fail(f"wheelhouse contains unpinned wheels: {', '.join(f'{name}=={version}' for name, version in sorted(extra))}")

    lines = list(HEADER)
    for name, version in pins:
        digest = hashes[(canonicalize_name(name), version)]
        lines.append(f"{name}=={version} {chr(92)}")
        lines.append(f"    --hash=sha256:{digest}")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({len(pins)} pins, {len(hashes)} hashes)")


if __name__ == "__main__":
    main()
