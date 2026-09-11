from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import RunConfig
from .errors import GoldMinerError
from .env import load_dotenv, project_dotenv
from .output.paths import default_output
from .pipeline import run_delivery
from .providers.analysis import OpenAIAnalysisProvider
from .providers.transcription import OpenAITranscriptionProvider
from .providers.scoring import OpenAIScoringProvider
from .providers.embeddings import OpenAIEmbeddingProvider
from .providers.reranking import OpenAIRerankingProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="goldminer",
        description="Discover, score, deduplicate, and rank short-form podcast clips.",
    )
    parser.add_argument("video", type=Path, help="source podcast video (never modified)")
    parser.add_argument("--output", type=Path, help="run directory (default: <video-folder>/<video-name>_clips)")
    parser.add_argument("--top-k", type=int, default=8, help="final ranked candidate count")
    parser.add_argument("--transcript", type=Path, help="timestamped transcript (.srt or .vtt)")
    parser.add_argument("--no-cut", action="store_true", help="write ranking, report, and manifest without cutting MP4 files")
    parser.add_argument("--handle-ms", type=int, default=500, help="extra media before and after each selected range (default: 500)")
    parser.add_argument("--max-duration", type=int, default=60, help="maximum final clip duration in seconds (default: 60)")
    parser.add_argument("--resume", action="store_true", help="reuse matching completed stage and clip artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        load_dotenv(project_dotenv())
    except GoldMinerError as exc:
        print(f"goldminer: error: {exc}", file=sys.stderr)
        return 2
    output = args.output or default_output(args.video)
    config = RunConfig(
        video=args.video,
        output=output,
        top_k=args.top_k,
        transcript=args.transcript,
        no_cut=args.no_cut,
        resume=args.resume,
        handle_ms=args.handle_ms,
        max_clip_seconds=args.max_duration,
    )
    try:
        provider = (
            None
            if config.transcript
            else OpenAITranscriptionProvider.from_environment(
                progress=lambda message: print(message, flush=True)
            )
        )
        result = run_delivery(
            config,
            transcription_provider=provider,
            analysis_provider=OpenAIAnalysisProvider.from_environment(),
            scoring_provider=OpenAIScoringProvider.from_environment(),
            embedding_provider=OpenAIEmbeddingProvider.from_environment(),
            reranking_provider=OpenAIRerankingProvider.from_environment(),
            progress=lambda message: print(message, flush=True),
        )
    except GoldMinerError as exc:
        print(f"goldminer: error: {exc}", file=sys.stderr)
        expected_log = output.expanduser().resolve() / "logs" / "run.jsonl"
        if expected_log.is_file():
            print(f"Diagnostic log: {expected_log}", file=sys.stderr)
        return 2

    ranking_result = result.ranking_result
    scoring_result = ranking_result.scoring_result
    discovery_result = scoring_result.discovery_result
    transcript_result = discovery_result.transcript_result
    foundation = transcript_result.foundation
    cache_note = " (reused)" if foundation.reused_audio else ""
    transcript_note = " (reused)" if transcript_result.reused_transcript else ""
    candidates_note = " (reused)" if discovery_result.reused_candidates else ""
    print("Clip cutting, report, and manifest complete")
    print(f"Source: {config.video.expanduser().resolve()}")
    print(f"Duration: {foundation.duration_ms / 1000:.3f} seconds")
    print(f"Audio: {foundation.paths.audio}{cache_note}")
    print(f"Transcript: {foundation.paths.transcript}{transcript_note}")
    print(f"Utterances: {len(transcript_result.transcript.utterances)}")
    print(f"Analysis windows: {discovery_result.window_count}")
    print(f"Candidates: {foundation.paths.candidates}{candidates_note}")
    print(f"Candidate count: {len(discovery_result.candidates)}")
    print(f"Scoring batches: {scoring_result.batch_count}")
    print(f"Scored candidates: {foundation.paths.scored_candidates}")
    print(f"Rejected candidates: {sum(item.rejected for item in scoring_result.scored_candidates)}")
    print(f"Ranking shortlist: {ranking_result.ranking.shortlist_count}")
    print(f"Final selections: {len(ranking_result.ranking.selected)}")
    print(f"Ranking: {foundation.paths.ranking}")
    print(f"Clips: {len(result.clips)}" + (" (--no-cut)" if config.no_cut else ""))
    print(f"Report: {foundation.paths.report}")
    print(f"Manifest: {foundation.paths.manifest}")
    print(f"Run directory: {foundation.paths.root}")
    print(f"Diagnostic log: {foundation.paths.run_log}")
    print("Next milestone: evaluation hardening")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
