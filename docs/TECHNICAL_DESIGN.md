# Milestones 1–7 Technical Design

## Scope

Milestone 1 provides an installable typed Python package and a `goldminer VIDEO` command. The command validates a regular input file, confirms FFmpeg and FFprobe are executable, creates a deterministic run directory, probes media duration, and atomically extracts mono 16 kHz PCM audio to `audio.wav`.

The source media is opened only for reading. Process commands are passed as argument arrays. A failed extraction removes its temporary artifact and leaves any prior valid output untouched.

## Components

- `goldminer.cli`: argument parsing, user-facing errors, and completion output
- `goldminer.config`: typed invocation configuration and validation
- `goldminer.output.paths`: output path resolution and containment checks
- `goldminer.media.ffmpeg`: executable discovery, process execution, and media probing
- `goldminer.media.audio`: cached audio validation and atomic extraction

## Canonical transcript foundation

Milestone 2 imports UTF-8 SRT and WebVTT transcripts into a versioned `transcript.json`. Every canonical utterance has a stable ordered ID, millisecond timestamps, an optional speaker, immutable source text, and separately normalized analysis text. Adjacent overlapping duplicate subtitle fragments collapse without rewriting their source wording.

Schema validation rejects empty content, malformed timestamps, unordered utterances, timestamps outside the media duration, duplicate IDs, and damaged cached JSON. Writes are atomic. Resume reuses a canonical transcript only when its source fingerprint and media duration still match.

Automatic transcription sits behind `TranscriptionProvider`; tests use `FakeTranscriptionProvider`. The CLI uses `OpenAITranscriptionProvider` when no transcript file is supplied. It requires `OPENAI_API_KEY`, splits analysis audio into sequential 10-minute 16 kHz mono 32 kbps MP3 chunks, requests timestamped speaker segments from `gpt-4o-transcribe-diarize`, offsets each response onto the full media timeline, validates every returned boundary, and deletes every temporary upload file. Successful per-chunk responses are fingerprinted and cached so interrupted runs do not repeat paid requests. Transient API and socket timeout failures are retried; credentials and transcript content are never logged.

Candidate analysis, ranking, cutting, reports, and manifests remain outside this milestone.

## Candidate discovery

Milestone 3 builds five-minute utterance-aligned windows with one minute of overlap. Each window is analyzed against all ten required detector categories in one structured OpenAI Responses API call. Provider results must cite start and end utterance IDs from the supplied window; deterministic code expands that range only through contiguous canonical utterances and derives transcript text and timestamps from the source.

Per-window responses are fingerprinted by transcript, model, prompt version, and window bounds. `--resume` reuses both window responses and a matching final `candidates.json`. Highly overlapping proposals merge only when their normalized core ideas are also similar. Scoring, boundary repair, semantic embedding deduplication, reranking, and final selection remain outside Milestone 3.

## Boundary optimization and scoring

Milestone 4 gives the scoring provider each raw candidate plus at least five neighboring utterances and up to 60 seconds of surrounding speech on both sides. The expanded context lets boundary repair consolidate a hook, development, and explicit payoff that candidate discovery split across adjacent proposals. The provider may trim or extend only inside that contiguous context. It selects the first and last utterance IDs plus verbatim phrase anchors inside those utterances. A phrase anchor may cross a provider-created segment boundary; deterministic matching uses the exact prefix or suffix contained in the named boundary utterance. Token alignment derives phrase-level timestamps by interpolation within the provider's timestamped segment and rebuilds only contiguous, exact source text. This lets a coherent sub-60-second arc survive when the transcription provider grouped its opening or payoff into a long utterance. The alignment never invents or rewrites speech.

Deterministic code computes the configured 100-point rubric, subtracts separately recorded penalties, and clamps results to 0–100. It rejects unsafe extraction, required non-contiguous splicing, a repaired duration outside 15–60 seconds, a score below 60, an opening that does not make sense to a cold viewer, or a weak/incomplete central idea. The provider must explicitly classify opening problems (`continuation`, `unresolved_reference`, `housekeeping`, or `unrelated_setup`) and central-idea problems (`generic`, `incomplete`, `garbled`, or `multiple_competing_ideas`); deterministic code rejects any non-`none` classification even if a separate summary flag is inconsistent. It also rejects known context-dependent opening patterns, grammatically unfinished endings, a missing explicit payoff quote, or an arc classified as incomplete. Standalone clarity, value, and payoff must each score at least 7. If no candidate clears these editorial gates, the run succeeds with zero clips instead of publishing weak material.

Candidates are evaluated one at a time using strict JSON Schema outputs. For every discovered idea the provider proposes three contiguous edits: `hook_first`, `concise`, and `full_arc`. One candidate per request ensures the model cannot silently omit later candidates and gives every completed idea an independent resume checkpoint. Deterministic optimization also recombines source-exact opening and ending anchors within an idea and across discovery candidates whose source ranges overlap by at least half. A candidate that fails only because it is no more than 15 seconds over the duration ceiling receives a deterministic setup-trimming pass using source-exact cold-opening anchors, including anchors later in the first coarse transcription utterance, while preserving its assessed payoff. Identical ranges with conflicting central-idea judgments are rejected conservatively. Expanded cold-opening gates reject conversational continuations, unresolved pronouns, contractions that refer backward, and deictic phrases, and vague action endings such as “go there” cannot serve as a payoff. Deterministic validation applies all duration, opening, central-idea, and narrative-arc gates to every version. Because recombination or duration rescue can materially change the content that was originally judged, any derived edit that would win receives a separate cached provider review of its exact retained transcript. That review rescored every dimension and independently rechecks its literal opening, single central idea, and matching payoff; a rejected derived winner falls back to the next candidate edit. The strongest publishable edit is then selected using total editorial score with hook and payoff tie-breakers. `scored_candidates.json` records the selected version and every alternative for auditing. Every raw score requires evidence from the exact text being judged. Initial provider responses are fingerprinted by candidate artifact, model, model-judgment version, and boundary version, while derived reviews are fingerprinted by their exact timestamps and transcript so paid judgments are safely reusable. Results are written atomically to `scored_candidates.json`. Deduplication, comparative reranking, confidence, and final diversity selection remain outside Milestone 4.

## Semantic deduplication and final ranking

Milestone 5 excludes hard-rejected candidates, embeds each remaining category-plus-transcript with `text-embedding-3-small`, and forms duplicate families using either at least 60% timestamp overlap or cosine similarity of at least 0.86. Each family retains the highest-scoring representative, with shorter duration and stable ID as deterministic tie-breakers.

The top 25 representatives enter one comparative structured-output reranking request. Final selection follows that order while limiting each primary category to two clips when alternatives exist, then fills remaining `--top-k` slots if needed. `ranking.json` records thresholds, provider versions, duplicate families and suppressed IDs, score and comparative ranks, rank agreement, confidence, and the reason for every selected clip. Embeddings and reranking are fingerprinted and cached under `ranking_cache/`.

## Clip cutting report and manifest

Milestone 6 cuts each final selection from the read-only source using FFmpeg argument arrays and accurate H.264/AAC re-encoding. A configurable `--handle-ms` defaults to up to 500 ms before and after the selected transcript range. Handles are clamped to media bounds, the configured duration ceiling, and transcript-detected silent gaps; a phrase-level boundary inside an utterance receives no handle on that side, preventing deliberately trimmed speech from leaking back into the clip. Final media is at most 60 seconds by default. Each clip is created under a temporary filename, checked with FFprobe against an explicit 750 ms duration tolerance, then atomically moved to a stable `RR_category_score.mp4` name. The default run directory is `<source-folder>/<video-name>_clips`.

Per-clip checkpoints bind the source SHA-256, candidate, timestamps, and cutter version. With `--resume`, a clip is reused only when its checkpoint matches and its measured duration remains valid. `--no-cut` skips MP4 creation while still producing the remaining artifacts. `report.html` is a standalone local review page with escaped transcript text and relative video links. `run_manifest.json` records the source fingerprint, configuration hash, algorithm and model versions, counts, timing, errors, artifact paths, and measured clip durations.

## Editorial evaluation and regression

Milestone 7 defines a versioned local evaluation dataset containing independently selected human moments, exact millisecond boundaries, 1–3 relevance, and semantic idea groups. Predictions and human moments match when their overlap covers at least 50 percent of the shorter range.

`goldminer-evaluate DATASET` reports Recall at 10, Precision at 5, NDCG at 10, mean boundary error, duplicate rate, and run reliability without calling live or paid providers. Reports are atomic versioned JSON. A prior report can be supplied as a baseline, and `--fail-on-regression` gives automated checks a nonzero exit status when a metric degrades beyond the configured tolerance.

## Hosted background-job adapter

The hosted adapter is a separate interface around the complete local pipeline; it does not change any discovery, boundary, scoring, ranking, or cutting behavior. A versioned job request identifies a source object in Cloudflare R2 and constrains `top_k`, handles, and final duration. Hosted requests cap clips at 60 seconds. The adapter downloads source media to an isolated ephemeral work directory, invokes the existing `python -m goldminer` entry point with a subprocess argument array, uploads final review artifacts and diagnostic logs to a job-scoped R2 prefix, and removes local working files.

The public v1 adapter creates a server-owned upload record and a short-lived, object-scoped R2 PUT URL. After R2 size and media-type verification, the upload ID resolves to the internal source key and can create exactly one workflow job. Frontend status responses normalize internal Workflow and container state without exposing provider or infrastructure errors. Clip listings return short-lived R2 GET URLs for direct browser playback, while the authenticated clip endpoint streams from the R2 binding and honors HTTP byte ranges. The legacy source-key API remains a trusted operational interface.

Cloudflare Workflows provides durable asynchronous orchestration. Each workflow is assigned deterministically to one of three Container processing slots and waits for its synchronous `/run` response, while the public API immediately returns a job ID. Each slot processes one job at a time and keeps every job in an isolated directory. A failed first attempt retains its local checkpoints for the Workflow retry; successful work is removed. The container image includes Python, FFmpeg, FFprobe, and the optional R2 client. Job state is also written to R2 so completion and failure information remains available independently of ephemeral container disk. Provider secrets and R2 credentials are passed to the container through Worker secrets and are never included in job payloads or result artifacts.
