import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_seed_patch.py"


def run_checker(scoring_path: Path, ban_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(scoring_path), str(ban_path)],
        capture_output=True,
        text=True,
    )


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_matching_patch_passes(tmp_path: Path) -> None:
    scoring = write_json(tmp_path / "scoring.json", {"patch_version": "7.41"})
    bans = write_json(
        tmp_path / "bans.json",
        {"rows": [{"patch_version": "7.41", "hero_id": 1, "bans": 2, "games": 9}]},
    )

    result = run_checker(scoring, bans)

    assert result.returncode == 0, result.stderr
    assert "patch 7.41" in result.stdout


def test_ban_seed_without_the_active_patch_fails(tmp_path: Path) -> None:
    scoring = write_json(tmp_path / "scoring.json", {"patch_version": "7.42"})
    bans = write_json(
        tmp_path / "bans.json",
        {"rows": [{"patch_version": "7.41", "hero_id": 1, "bans": 2, "games": 9}]},
    )

    result = run_checker(scoring, bans)

    assert result.returncode == 1
    assert "no ban rows for patch 7.42" in result.stderr
    assert "7.41" in result.stderr


def test_missing_active_patch_fails(tmp_path: Path) -> None:
    scoring = write_json(tmp_path / "scoring.json", {})
    bans = write_json(
        tmp_path / "bans.json",
        {"rows": [{"patch_version": "7.41", "hero_id": 1, "bans": 2, "games": 9}]},
    )

    result = run_checker(scoring, bans)

    assert result.returncode == 1
    assert "no patch_version" in result.stderr
