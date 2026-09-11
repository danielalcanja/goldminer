from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from .models import Utterance

_SPACE_RE = re.compile(r"\s+")
_SPEAKER_RE = re.compile(r"^([\w][\w .'-]{0,48}):\s+(.+)$", re.DOTALL)
_VTT_VOICE_RE = re.compile(r"^<v(?:\.[^ >]+)*(?:\s+([^>]+))?>(.*?)</v>$", re.DOTALL | re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True, slots=True)
class RawCue:
    start_ms: int
    end_ms: int
    text: str
    speaker: str | None = None


def clean_cue_text(text: str, speaker: str | None = None) -> tuple[str | None, str]:
    value = html.unescape(text).strip()
    voice_match = _VTT_VOICE_RE.match(value)
    if voice_match:
        speaker = speaker or (voice_match.group(1).strip() if voice_match.group(1) else None)
        value = voice_match.group(2)
    value = _TAG_RE.sub("", value)
    value = _SPACE_RE.sub(" ", value).strip()
    speaker_match = _SPEAKER_RE.match(value)
    if speaker is None and speaker_match:
        speaker = speaker_match.group(1).strip()
        value = speaker_match.group(2).strip()
    return speaker, value


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    return _SPACE_RE.sub(" ", value).strip()


def canonicalize_cues(cues: Iterable[RawCue]) -> tuple[Utterance, ...]:
    ordered = sorted(cues, key=lambda cue: (cue.start_ms, cue.end_ms))
    retained: list[RawCue] = []
    for cue in ordered:
        speaker, original = clean_cue_text(cue.text, cue.speaker)
        if not original:
            continue
        cleaned = RawCue(cue.start_ms, cue.end_ms, original, speaker)
        if retained:
            previous = retained[-1]
            same_words = normalize_text(previous.text).casefold() == normalize_text(cleaned.text).casefold()
            overlaps = cleaned.start_ms < previous.end_ms
            if same_words and overlaps and previous.speaker == cleaned.speaker:
                retained[-1] = RawCue(
                    start_ms=min(previous.start_ms, cleaned.start_ms),
                    end_ms=max(previous.end_ms, cleaned.end_ms),
                    text=previous.text,
                    speaker=previous.speaker,
                )
                continue
        retained.append(cleaned)

    return tuple(
        Utterance(
            id=f"u{index:06d}",
            start_ms=cue.start_ms,
            end_ms=cue.end_ms,
            speaker=cue.speaker,
            original_text=cue.text,
            normalized_text=normalize_text(cue.text),
        )
        for index, cue in enumerate(retained, start=1)
    )
