# Podcast Gold Miner Product Specification

## Product goal

Given a 20–90 minute timestamped podcast transcript, return roughly 5–15 coherent, explainable, cut-ready moments suitable for short-form platforms. A selected moment must have a natural opening, development, payoff, exact source timestamps, and no fabricated or non-contiguous transcript content.

## Local command

```text
goldminer podcast.mp4 [--output DIR] [--top-k 8] [--transcript PATH] [--no-cut] [--resume]
```

The complete local pipeline validates its environment, extracts analysis audio, imports or creates a canonical transcript, discovers candidates, repairs boundaries, scores and deduplicates candidates, reranks a diverse final set, writes JSON and HTML artifacts, and optionally cuts MP4 clips.

## Staged delivery

1. CLI and media foundation
2. Canonical transcript
3. Candidate discovery
4. Boundary optimization and scoring
5. Deduplication and final ranking
6. Clip cutting, report, manifest, and resume
7. Evaluation hardening

Each milestone must be tested before the next begins. The supplied DOCX remains the complete product authority; this file records the implementation-facing summary.

Milestone 7 uses a versioned local human-label dataset and a local evaluation command to report Recall at 10, Precision at 5, NDCG at 10, boundary error, duplicate rate, and run reliability. Baseline comparison can fail when an algorithm change regresses an agreed metric.
