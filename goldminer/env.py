from __future__ import annotations

import os
from pathlib import Path

from .errors import GoldMinerError


def _decode_value(raw: str, line_number: int) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {'"', "'"}:
        quote = value[0]
        if len(value) < 2 or value[-1] != quote:
            raise GoldMinerError(f"Malformed quoted value in .env line {line_number}")
        value = value[1:-1]
        if quote == '"':
            value = value.replace("\\n", "\n").replace("\\\"", '"').replace("\\\\", "\\")
        return value
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value


def load_dotenv(path: Path) -> bool:
    """Load a small, predictable .env format without overriding shell values."""
    if not path.exists():
        return False
    if not path.is_file():
        raise GoldMinerError(f".env path is not a regular file: {path}")
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        raise GoldMinerError(f"Could not read .env file: {exc}") from exc

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise GoldMinerError(f"Malformed .env line {line_number}; expected NAME=value")
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "a").isalnum() or name[0].isdigit():
            raise GoldMinerError(f"Invalid variable name in .env line {line_number}")
        os.environ.setdefault(name, _decode_value(raw_value, line_number))
    return True


def project_dotenv() -> Path:
    return Path(__file__).resolve().parent.parent / ".env"
