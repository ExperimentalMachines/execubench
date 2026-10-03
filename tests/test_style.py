"""House style: no em dashes or en dashes anywhere in our own text."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DASHES = (chr(0x2014), chr(0x2013))  # em dash, en dash; written as codes so this file passes
TEXT = {".md", ".py", ".json", ".yaml", ".yml", ".sh", ".toml", ".txt"}
SKIP = {"data", ".git", ".venv", ".pytest_cache", ".ruff_cache"}


def test_no_em_or_en_dashes():
    offenders = []
    for path in ROOT.rglob("*"):
        if path.suffix not in TEXT or any(part in SKIP for part in path.relative_to(ROOT).parts):
            continue
        text = path.read_text(errors="replace")
        if any(d in text for d in DASHES):
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
