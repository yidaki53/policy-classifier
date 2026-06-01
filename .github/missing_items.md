# Gap analysis — missing items to complete project

This document summarises what remains to be done for a fully reproducible, packaged,
and publishable policy-classifier pipeline. It draws on an inspection of the repository
and the `agent_plan.md` roadmap.

High-priority items
-------------------
- Move the runtime package into `src/` and add packaging metadata (`pyproject.toml` or `setup.cfg`) so
  the project can be installed with `pip install -e .`. This will make imports robust and ease CI.
- Add GitHub Actions CI workflow to run `pytest`, `definitions.loader --verify`, and a smoke run of `scripts/ingest.py --no-sample` (or sample mode) to ensure end-to-end reproducibility.
- Add a `scripts/verify.py` (recommended) that runs a deterministic sample ingestion, classification, aggregation,
  and asserts checksums & expected outputs (used by CI).
- Add lineage entries for normalization: currently `scripts/classify.py` inserts into `normalized_motions` but does
  not call `record_lineage` for those normalization inserts — add that for full provenance.

Medium-priority items
---------------------
- Provide an editable install (`pyproject.toml`) and a minimal `Makefile` (or update `manuscript/Makefile`) that
  supports `make env`/`make deps` to reproduce environment.
- Add a lightweight Dockerfile (optional) to support fully reproducible runs in CI or for reviewers.
- Add explicit configuration for hyperparameters and runtime options (recency lambda, embedding weight,
  supervised thresholds) via a central `config/` or `pyproject` section, and use them in scripts.
- Add more unit/integration tests for supervised fallback, embedding matcher, and the recency-weight math.

Low-priority / polishing
-------------------------
- Move large binary artifacts (trained models) out of code tree into `data/models/` and add download script.
- Add code to compute and store checksums for outputs/figures and optionally add those to `lineage`.
- Improve documentation: add `docs/` or expand README with quickstart, dev, and publishing instructions.

Quick wins (I can do next)
-------------------------
1. Add `.github/directory_tree.md` (done) and reference it from `.github/copilot-instructions.md` (done).
2. Create a small CI workflow that runs `pytest` on push and checks `definitions/` checksum.
3. Add a `scripts/verify.py` smoke test that the pipeline runs on sample data.

If you'd like, I can implement any of the above items now — tell me whether to (a) perform the `src/` move
    now, including adjusting imports and packaging files, or (b) add CI + verify script first, or (c) start
    by adding `pyproject.toml` for editable installs and leave the code in place.
