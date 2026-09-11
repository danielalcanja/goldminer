from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
from typing import Any, Sequence

from goldminer.analysis.ranking import DEDUPE_VERSION, SELECTION_VERSION
from goldminer.analysis.scoring import BOUNDARY_VERSION, SCORING_VERSION
from goldminer.output.cutter import CUTTER_VERSION, ClipCut

MANIFEST_SCHEMA_VERSION = "1.0"
REPORT_VERSION = "1.0"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_report(path: Path, selections: Sequence[dict[str, Any]], clips: Sequence[ClipCut]) -> None:
    clip_by_id = {clip.candidate_id: clip for clip in clips}
    cards = []
    for item in selections:
        clip = clip_by_id.get(str(item["id"]))
        media = f'<video controls preload="metadata" src="clips/{html.escape(clip.path.name)}"></video>' if clip else '<p class="not-cut">Video cutting was skipped.</p>'
        transcript = html.escape(str(item["transcript"]))
        cards.append(f"""<article><div class="rank">#{item['rank']}</div><h2>{html.escape(str(item['category']).replace('_', ' ').title())}</h2>
<div class="meta">Score {item['final_score']} · {item['duration_sec']} sec · {html.escape(str(item['confidence']).title())} confidence</div>
{media}<h3>Why it ranked</h3><p>{html.escape(str(item['comparative_reason']))}</p>
<h3>Transcript</h3><p class="transcript">{transcript}</p>
<div class="time">{item['start_ms']}–{item['end_ms']} ms · {html.escape(str(item['id']))}</div></article>""")
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Podcast Gold Miner Report</title><style>
:root{{--ink:#17202a;--muted:#607080;--gold:#c58b18;--paper:#f6f4ef}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}header{{background:#182634;color:white;padding:48px max(5vw,24px)}}header h1{{margin:0;font-size:clamp(30px,5vw,52px)}}header p{{color:#cfdae3;max-width:700px}}main{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:24px;padding:32px max(5vw,24px)}}article{{position:relative;background:white;padding:28px;border-radius:14px;box-shadow:0 4px 18px #17202a14}}.rank{{position:absolute;right:24px;top:20px;color:var(--gold);font-size:24px;font-weight:800}}h2{{margin:0 48px 4px 0}}h3{{margin:22px 0 6px;font-size:15px;text-transform:uppercase;letter-spacing:.06em}}.meta,.time{{color:var(--muted);font-size:14px}}video{{width:100%;margin-top:20px;border-radius:8px;background:#111}}.transcript{{white-space:pre-wrap}}.not-cut{{padding:18px;background:#eef1f3;border-radius:8px}}footer{{padding:20px max(5vw,24px);color:var(--muted)}}
</style></head><body><header><h1>Podcast Gold Miner Report</h1><p>{len(selections)} ranked, contiguous moments selected from the source podcast.</p></header><main>{''.join(cards)}</main><footer>Generated locally by Podcast Gold Miner · Report schema {REPORT_VERSION}</footer></body></html>"""
    _atomic_text(path, document)


def write_manifest(path: Path, *, source: Path, source_sha256: str, duration_ms: int, config: dict[str, Any], counts: dict[str, int], models: dict[str, str], artifacts: Sequence[str], clips: Sequence[ClipCut], no_cut: bool, timings_ms: dict[str, int] | None = None) -> None:
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    value = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "status": "completed",
        "algorithm_version": "goldminer-v0.1.0",
        "input": {"path": str(source), "sha256": source_sha256, "duration_ms": duration_ms},
        "algorithm_versions": {"boundary": BOUNDARY_VERSION, "scoring": SCORING_VERSION, "dedupe": DEDUPE_VERSION, "selection": SELECTION_VERSION, "cutter": CUTTER_VERSION, "report": REPORT_VERSION},
        "models": models,
        "config": config,
        "config_hash": config_hash,
        "counts": counts,
        "timings_ms": timings_ms or {},
        "errors": [],
        "no_cut": no_cut,
        "artifacts": list(artifacts),
        "clips": [{"rank": clip.rank, "candidate_id": clip.candidate_id, "path": str(clip.path), "start_ms": clip.start_ms, "end_ms": clip.end_ms, "measured_duration_ms": clip.measured_duration_ms, "reused": clip.reused} for clip in clips],
    }
    _atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")
