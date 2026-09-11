# Gold Miner Engineering Instructions

Gold Miner identifies the highest-value short-form video moments from long-form podcasts and cuts those moments from the original local video.

## Read first

Before changing behavior, read `docs/PRODUCT_SPEC.md` and `docs/TECHNICAL_DESIGN.md`. If implementation and documentation conflict, stop and describe the conflict.

## Version 1 boundaries

- Keep the application local-first with `goldminer VIDEO` as its primary interface.
- Store run state in local JSON files and directories.
- Never overwrite or delete source media.
- Keep external integrations behind interfaces and use fakes in tests.
- Preserve exact source text and contiguous timestamp ranges in later milestones.
- Use typed Python, millisecond timestamps, subprocess argument arrays, atomic writes, and actionable errors.
- Implement and verify one product-spec milestone at a time.
