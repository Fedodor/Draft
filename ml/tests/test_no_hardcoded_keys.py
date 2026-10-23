import re
from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1] / "training"

PATTERNS = {
    "key assignment": re.compile(
        r"(?i)\b(?:api_key|stratz_key|opendota_key|stratz_key_default|opendota_key_default)\b"
        r"\s*=\s*['\"][^'\"]+['\"]"
    ),
    "JWT literal": re.compile(r"eyJhbGciOi"),
}


def test_training_sources_contain_no_hardcoded_credentials():
    offenders = []
    for path in TRAINING_ROOT.glob("*.py"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in PATTERNS.items():
            if pattern.search(text):
                offenders.append(f"{path.name}: {label}")

    assert not offenders, "Hardcoded credentials found: " + ", ".join(offenders)
