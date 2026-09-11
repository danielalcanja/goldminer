from __future__ import annotations

from dataclasses import dataclass

from .config import RunConfig
from .media.audio import extract_analysis_audio
from .media.ffmpeg import discover_tools, probe_duration_ms, validate_tools
from .output.paths import RunPaths, create_run_paths
from .output.runlog import RunLogger
from .providers.transcription import TranscriptionProvider
from .transcript.artifact import build_transcript
from .transcript.models import CanonicalTranscript
from .analysis.artifact import (
    CachedAnalysisProvider,
    file_sha256,
    load_candidates_artifact,
    write_candidates_artifact,
)
from .analysis.discovery import discover_candidates
from .analysis.models import Candidate
from .analysis.windows import build_discovery_windows
from .providers.analysis import AnalysisProvider
from .analysis.boundaries import build_boundary_context
from .analysis.scored_artifact import (
    CachedScoringProvider,
    SCORING_BATCH_SIZE,
    score_candidates,
    write_scored_artifact,
)
from .analysis.scoring import ScoredCandidate
from .providers.scoring import ScoringProvider
from .analysis.ranking import RankingResult, build_ranking, write_ranking
from .providers.embeddings import EmbeddingProvider
from .providers.reranking import RerankingProvider
from .output.cutter import ClipCut, cut_selected_clips
from .output.delivery import file_sha256 as source_file_sha256, write_manifest, write_report
import json
from time import perf_counter


@dataclass(frozen=True, slots=True)
class FoundationResult:
    paths: RunPaths
    duration_ms: int
    reused_audio: bool


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    foundation: FoundationResult
    transcript: CanonicalTranscript
    reused_transcript: bool


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    transcript_result: TranscriptResult
    candidates: tuple[Candidate, ...]
    window_count: int
    reused_candidates: bool


@dataclass(frozen=True, slots=True)
class ScoringResult:
    discovery_result: DiscoveryResult
    scored_candidates: tuple[ScoredCandidate, ...]
    batch_count: int


@dataclass(frozen=True, slots=True)
class FinalRankingResult:
    scoring_result: ScoringResult
    ranking: RankingResult


@dataclass(frozen=True, slots=True)
class DeliveryResult:
    ranking_result: FinalRankingResult
    clips: tuple[ClipCut, ...]


def _run_media_foundation(config: RunConfig, paths: RunPaths, logger: RunLogger) -> FoundationResult:
    started = logger.start("media_foundation", source_path=str(config.video))
    try:
        tools = discover_tools()
        validate_tools(tools)
        logger.write(
            "media_tools",
            "validated",
            ffmpeg_path=tools.ffmpeg,
            ffprobe_path=tools.ffprobe,
        )
        duration_ms = probe_duration_ms(config.video, tools)
        reused = extract_analysis_audio(config.video, paths.audio, tools, resume=config.resume)
        logger.complete(
            "media_foundation",
            started,
            duration_media_ms=duration_ms,
            audio_path=str(paths.audio),
            audio_bytes=paths.audio.stat().st_size,
            reused_audio=reused,
        )
        return FoundationResult(paths=paths, duration_ms=duration_ms, reused_audio=reused)
    except BaseException as exc:
        logger.failure("media_foundation", started, exc)
        raise


def _prepare_run(config: RunConfig) -> tuple[RunConfig, RunPaths, RunLogger]:
    checked = config.validate()
    paths = create_run_paths(checked.output, checked.video)
    logger = RunLogger(paths.run_log)
    logger.write(
        "run",
        "started",
        **logger.environment(),
        source_path=str(checked.video),
        output_path=str(paths.root),
        transcript_path=str(checked.transcript) if checked.transcript else None,
        top_k=checked.top_k,
        resume=checked.resume,
        no_cut=checked.no_cut,
    )
    return checked, paths, logger


def run_media_foundation(config: RunConfig) -> FoundationResult:
    checked, paths, logger = _prepare_run(config)
    try:
        result = _run_media_foundation(checked, paths, logger)
        logger.write("run", "completed", log_path=str(paths.run_log))
        return result
    except BaseException as exc:
        logger.write("run", "failed", error_type=type(exc).__name__, error=str(exc))
        raise


def run_transcript_foundation(
    config: RunConfig,
    *,
    provider: TranscriptionProvider | None = None,
) -> TranscriptResult:
    checked, paths, logger = _prepare_run(config)
    try:
        foundation = _run_media_foundation(checked, paths, logger)
        started = logger.start(
            "transcript_foundation",
            source_type="file" if checked.transcript else "provider",
            provider=provider.name if provider else None,
            provider_model=provider.model if provider else None,
        )
        try:
            transcript, reused = build_transcript(
                foundation.paths.transcript,
                media_duration_ms=foundation.duration_ms,
                transcript_path=checked.transcript,
                provider=provider,
                audio_path=foundation.paths.audio,
                resume=checked.resume,
            )
        except BaseException as exc:
            logger.failure("transcript_foundation", started, exc)
            raise
        logger.complete(
            "transcript_foundation",
            started,
            transcript_json=str(paths.transcript),
            source_type=transcript.source_type,
            source_sha256=transcript.source_sha256,
            utterance_count=len(transcript.utterances),
            reused_transcript=reused,
        )
        logger.write("run", "completed", log_path=str(paths.run_log))
        return TranscriptResult(foundation=foundation, transcript=transcript, reused_transcript=reused)
    except BaseException as exc:
        logger.write("run", "failed", error_type=type(exc).__name__, error=str(exc), log_path=str(paths.run_log))
        raise


def run_candidate_discovery(
    config: RunConfig,
    *,
    transcription_provider: TranscriptionProvider | None = None,
    analysis_provider: AnalysisProvider,
    progress=None,
) -> DiscoveryResult:
    transcript_result = run_transcript_foundation(config, provider=transcription_provider)
    paths = transcript_result.foundation.paths
    logger = RunLogger(paths.run_log)
    started = logger.start(
        "candidate_discovery",
        provider=analysis_provider.name,
        provider_model=analysis_provider.model,
    )
    transcript_hash = file_sha256(paths.transcript)
    try:
        if config.resume and paths.candidates.is_file():
            metadata, cached_candidates = load_candidates_artifact(paths.candidates)
            if (
                metadata.get("transcript_sha256") == transcript_hash
                and metadata.get("model") == analysis_provider.model
            ):
                logger.complete(
                    "candidate_discovery",
                    started,
                    window_count=int(metadata["window_count"]),
                    candidate_count=len(cached_candidates),
                    reused_candidates=True,
                )
                return DiscoveryResult(
                    transcript_result=transcript_result,
                    candidates=cached_candidates,
                    window_count=int(metadata["window_count"]),
                    reused_candidates=True,
                )
        windows = build_discovery_windows(transcript_result.transcript.utterances)
        cached_provider = CachedAnalysisProvider(analysis_provider, paths.analysis_cache, transcript_hash)
        candidates = discover_candidates(
            transcript_result.transcript, windows, cached_provider, progress=progress
        )
        write_candidates_artifact(
            paths.candidates,
            transcript_sha256=transcript_hash,
            provider=analysis_provider,
            window_count=len(windows),
            candidates=candidates,
        )
        logger.complete(
            "candidate_discovery",
            started,
            window_count=len(windows),
            candidate_count=len(candidates),
            candidates_json=str(paths.candidates),
            reused_candidates=False,
        )
        return DiscoveryResult(
            transcript_result=transcript_result,
            candidates=candidates,
            window_count=len(windows),
            reused_candidates=False,
        )
    except BaseException as exc:
        logger.failure("candidate_discovery", started, exc)
        raise


def run_boundary_scoring(
    config: RunConfig,
    *,
    transcription_provider: TranscriptionProvider | None,
    analysis_provider: AnalysisProvider,
    scoring_provider: ScoringProvider,
    progress=None,
) -> ScoringResult:
    discovery_result = run_candidate_discovery(
        config,
        transcription_provider=transcription_provider,
        analysis_provider=analysis_provider,
        progress=progress,
    )
    paths = discovery_result.transcript_result.foundation.paths
    logger = RunLogger(paths.run_log)
    started = logger.start(
        "boundary_scoring",
        provider=scoring_provider.name,
        provider_model=scoring_provider.model,
        candidate_count=len(discovery_result.candidates),
    )
    try:
        contexts = tuple(
            build_boundary_context(
                candidate,
                discovery_result.transcript_result.transcript,
                padding_ms=60_000,
            )
            for candidate in discovery_result.candidates
        )
        candidates_hash = file_sha256(paths.candidates)
        cached_provider = CachedScoringProvider(
            scoring_provider, paths.scoring_cache, candidates_hash
        )
        scored = score_candidates(
            contexts,
            discovery_result.transcript_result.transcript,
            cached_provider,
            progress=progress,
            max_duration_ms=config.max_clip_seconds * 1000,
        )
        write_scored_artifact(
            paths.scored_candidates,
            candidates_sha256=candidates_hash,
            provider=scoring_provider,
            candidates=scored,
        )
        batch_count = (len(contexts) + SCORING_BATCH_SIZE - 1) // SCORING_BATCH_SIZE
        logger.complete(
            "boundary_scoring",
            started,
            batch_count=batch_count,
            scored_count=len(scored),
            rejected_count=sum(item.rejected for item in scored),
            scored_candidates_json=str(paths.scored_candidates),
        )
        return ScoringResult(discovery_result, scored, batch_count)
    except BaseException as exc:
        logger.failure("boundary_scoring", started, exc)
        raise


def run_final_ranking(
    config: RunConfig,
    *,
    transcription_provider: TranscriptionProvider | None,
    analysis_provider: AnalysisProvider,
    scoring_provider: ScoringProvider,
    embedding_provider: EmbeddingProvider,
    reranking_provider: RerankingProvider,
    progress=None,
) -> FinalRankingResult:
    scoring_result = run_boundary_scoring(
        config,
        transcription_provider=transcription_provider,
        analysis_provider=analysis_provider,
        scoring_provider=scoring_provider,
        progress=progress,
    )
    paths = scoring_result.discovery_result.transcript_result.foundation.paths
    logger = RunLogger(paths.run_log)
    started = logger.start("final_ranking", embedding_model=embedding_provider.model, reranking_model=reranking_provider.model)
    try:
        if progress:
            progress("Final ranking: embedding and deduplicating candidates")
        ranking, artifact = build_ranking(
            scoring_result.scored_candidates,
            embedding_provider,
            reranking_provider,
            paths.ranking_cache,
            config.top_k,
        )
        write_ranking(paths.ranking, artifact)
        logger.complete(
            "final_ranking", started, ranking_json=str(paths.ranking),
            shortlist_count=ranking.shortlist_count, selected_count=len(ranking.selected),
            reused_embeddings=ranking.reused_embeddings, reused_reranking=ranking.reused_reranking,
        )
        if progress:
            progress("Final ranking: completed")
        return FinalRankingResult(scoring_result, ranking)
    except BaseException as exc:
        logger.failure("final_ranking", started, exc)
        raise


def run_delivery(
    config: RunConfig,
    *,
    transcription_provider: TranscriptionProvider | None,
    analysis_provider: AnalysisProvider,
    scoring_provider: ScoringProvider,
    embedding_provider: EmbeddingProvider,
    reranking_provider: RerankingProvider,
    progress=None,
) -> DeliveryResult:
    ranking_result = run_final_ranking(
        config,
        transcription_provider=transcription_provider,
        analysis_provider=analysis_provider,
        scoring_provider=scoring_provider,
        embedding_provider=embedding_provider,
        reranking_provider=reranking_provider,
        progress=progress,
    )
    scoring_result = ranking_result.scoring_result
    discovery = scoring_result.discovery_result
    foundation = discovery.transcript_result.foundation
    paths = foundation.paths
    logger = RunLogger(paths.run_log)
    started = logger.start("delivery", no_cut=config.no_cut, selected_count=len(ranking_result.ranking.selected))
    try:
        ranking_artifact = json.loads(paths.ranking.read_text(encoding="utf-8"))
        selections = tuple(ranking_artifact["final_ranking"])
        if progress:
            progress("Delivery: fingerprinting source video")
        source_sha256 = source_file_sha256(config.video)
        clips: tuple[ClipCut, ...] = ()
        if not config.no_cut:
            tools = discover_tools()
            clips = cut_selected_clips(
                config.video, selections, paths.clips, tools,
                source_sha256=source_sha256,
                media_duration_ms=foundation.duration_ms,
                handle_ms=config.handle_ms,
                max_duration_ms=config.max_clip_seconds * 1000,
                utterances=discovery.transcript_result.transcript.utterances,
                resume=config.resume,
                progress=progress,
            )
        if progress:
            progress("Delivery: writing report and manifest")
        write_report(paths.report, selections, clips)
        artifacts = ["transcript.json", "candidates.json", "scored_candidates.json", "ranking.json", "report.html"]
        if clips:
            artifacts.append("clips/")
        counts = dict(ranking_artifact["counts"])
        counts.update({"utterances": len(discovery.transcript_result.transcript.utterances), "clips": len(clips)})
        write_manifest(
            paths.manifest,
            source=config.video,
            source_sha256=source_sha256,
            duration_ms=foundation.duration_ms,
            config={"top_k": config.top_k, "handle_ms": config.handle_ms, "max_clip_seconds": config.max_clip_seconds, "no_cut": config.no_cut, "resume": config.resume},
            counts=counts,
            models={
                "transcription": transcription_provider.model if transcription_provider else "imported-transcript",
                "analysis": analysis_provider.model,
                "scoring": scoring_provider.model,
                "embedding": embedding_provider.model,
                "reranking": reranking_provider.model,
            },
            artifacts=artifacts,
            clips=clips,
            no_cut=config.no_cut,
            timings_ms={"delivery": round((perf_counter() - started) * 1000)},
        )
        logger.complete("delivery", started, clip_count=len(clips), report_html=str(paths.report), manifest_json=str(paths.manifest))
        return DeliveryResult(ranking_result, clips)
    except BaseException as exc:
        logger.failure("delivery", started, exc)
        raise
