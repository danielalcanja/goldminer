from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from goldminer.errors import GoldMinerError


@dataclass(frozen=True, slots=True)
class RunPaths:
    root: Path
    audio: Path
    transcript: Path
    candidates: Path
    analysis_cache: Path
    scored_candidates: Path
    scoring_cache: Path
    ranking: Path
    ranking_cache: Path
    report: Path
    manifest: Path
    run_log: Path
    logs: Path
    clips: Path


def default_output(video: Path, cwd: Path | None = None) -> Path:
    expanded = video.expanduser()
    source = (cwd / expanded).resolve() if cwd is not None and not expanded.is_absolute() else expanded.resolve()
    return source.parent / f"{source.stem}_clips"


def create_run_paths(output: Path, source: Path) -> RunPaths:
    root = output.resolve()
    if root == source.resolve() or source.resolve().is_relative_to(root):
        raise GoldMinerError("Output directory cannot be the source video or contain the source video")
    if root.exists() and not root.is_dir():
        raise GoldMinerError(f"Output path exists and is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    clips = root / "clips"
    logs.mkdir(exist_ok=True)
    clips.mkdir(exist_ok=True)
    return RunPaths(
        root=root,
        audio=root / "audio.wav",
        transcript=root / "transcript.json",
        candidates=root / "candidates.json",
        analysis_cache=root / "analysis_cache",
        scored_candidates=root / "scored_candidates.json",
        scoring_cache=root / "scoring_cache",
        ranking=root / "ranking.json",
        ranking_cache=root / "ranking_cache",
        report=root / "report.html",
        manifest=root / "run_manifest.json",
        run_log=logs / "run.jsonl",
        logs=logs,
        clips=clips,
    )
