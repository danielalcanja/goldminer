from __future__ import annotations

import re
from pathlib import Path

from goldminer.errors import GoldMinerError

from .models import Utterance
from .normalize import RawCue, canonicalize_cues

_TIMING_RE = re.compile(
    r"^(?P<start>\d{1,2}:\d{2}:\d{2}[,.]\d{3})\s+-->\s+"
    r"(?P<end>\d{1,2}:\d{2}:\d{2}[,.]\d{3})(?:\s+.*)?$"
)


def parse_timestamp(value: str) -> int:
    normalized = value.replace(",", ".")
    parts = normalized.split(":")
    if len(parts) != 3 or "." not in parts[2]:
        raise GoldMinerError(f"Malformed transcript timestamp: {value}")
    hours_text, minutes_text, seconds_ms = parts
    seconds_text, milliseconds_text = seconds_ms.split(".", 1)
    try:
        hours = int(hours_text)
        minutes = int(minutes_text)
        seconds = int(seconds_text)
        milliseconds = int(milliseconds_text)
    except ValueError as exc:
        raise GoldMinerError(f"Malformed transcript timestamp: {value}") from exc
    if minutes > 59 or seconds > 59 or len(milliseconds_text) != 3:
        raise GoldMinerError(f"Malformed transcript timestamp: {value}")
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds


def _parse_blocks(text: str, *, is_vtt: bool) -> tuple[Utterance, ...]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = re.split(r"\n\s*\n", normalized)
    cues: list[RawCue] = []
    for block_number, block in enumerate(blocks, start=1):
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        upper = lines[0].upper()
        if is_vtt and (upper == "WEBVTT" or upper.startswith(("NOTE", "STYLE", "REGION"))):
            continue
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            if is_vtt and upper.startswith("WEBVTT"):
                continue
            raise GoldMinerError(f"Transcript block {block_number} has no timing line")
        match = _TIMING_RE.match(lines[timing_index])
        if not match:
            raise GoldMinerError(f"Malformed timing line in block {block_number}: {lines[timing_index]}")
        start_ms = parse_timestamp(match.group("start"))
        end_ms = parse_timestamp(match.group("end"))
        if end_ms <= start_ms:
            raise GoldMinerError(f"Transcript block {block_number} must end after it starts")
        cue_text = " ".join(lines[timing_index + 1 :]).strip()
        if not cue_text:
            raise GoldMinerError(f"Transcript block {block_number} has no text")
        cues.append(RawCue(start_ms=start_ms, end_ms=end_ms, text=cue_text))
    utterances = canonicalize_cues(cues)
    if not utterances:
        raise GoldMinerError("Transcript contains no timed utterances")
    return utterances


def parse_srt(text: str) -> tuple[Utterance, ...]:
    return _parse_blocks(text, is_vtt=False)


def parse_vtt(text: str) -> tuple[Utterance, ...]:
    return _parse_blocks(text, is_vtt=True)


def parse_transcript_file(path: Path) -> tuple[str, tuple[Utterance, ...]]:
    suffix = path.suffix.casefold()
    if suffix not in {".srt", ".vtt"}:
        raise GoldMinerError(f"Unsupported transcript format '{suffix or '(none)'}'; use .srt or .vtt")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise GoldMinerError(f"Could not read transcript {path}: {exc}") from exc
    if suffix == ".srt":
        return "srt", parse_srt(text)
    if suffix == ".vtt":
        return "vtt", parse_vtt(text)
    raise AssertionError("validated transcript suffix was not handled")
