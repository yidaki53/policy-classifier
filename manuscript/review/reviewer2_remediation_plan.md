# Reviewer #2 Remediation Plan

## Objective
Address major-review weaknesses with a rigorous, auditable remediation workflow that improves inferential discipline, methodological reproducibility, and manuscript submission readiness.

## Rigorous remediation protocol

1. Claim discipline and scope alignment
- Replace any wording that implies validated latent-trait recovery with bounded measurement language consistent with current classifier quality.
- Enforce non-causal framing and explicit uncertainty boundaries where substantive claims are made.
- Acceptance criterion: no section claims stronger inference than supported by current diagnostics.

2. Terminology and interpretation controls
- Add a concise definition block for ideology index, consistency, fulfillment, and contradiction so reviewers can audit interpretation quickly.
- Tie each defined construct to a specific output artifact family under output/analysis.
- Acceptance criterion: key constructs are defined once in methods/results and used consistently thereafter.

3. Deterministic analytics outputs
- Remove nondeterministic class-selection behavior in figure generation by using deterministic top-class extraction with explicit tie-breaking.
- Acceptance criterion: repeated figure runs on unchanged input produce identical category selections.

4. Reproducible write safety for long runs
- Harden parquet writer behavior with a lock-based critical section and atomic file replacement.
- Keep resume semantics explicit and auditable for speech-level outputs.
- Acceptance criterion: concurrent writes do not silently corrupt outputs; interruption leaves either old file or fully written new file.

5. Build and evidence verification
- Re-render manuscript sections and run targeted tests for deterministic extraction and writer safety.
- Re-run manuscript validation checks for publication-facing sections.
- Acceptance criterion: targeted tests pass and manuscript build/check pipeline succeeds.

## Implementation status (current cycle)

1. Claim discipline and scope alignment
- Status: completed
- Implemented in: manuscript/sections/02_question.md

2. Terminology and interpretation controls
- Status: completed
- Implemented in: manuscript/sections/03_methodology.md

3. Deterministic analytics outputs
- Status: completed
- Implemented in: scripts/generate_figures.py
- Regression test: tests/test_generate_figures.py

4. Reproducible write safety for long runs
- Status: completed
- Implemented in: src/swedish_parliament_policy_classifier/classifier/persistence_port.py
- Regression test: tests/test_parquet_writer.py

5. Quantitative claim synchronization in manuscript
- Status: completed
- Implemented in: manuscript/sections/03_results.md
- Notes: hardcoded corpus/evaluation/linkage count claims replaced by rendered context blocks to reduce stale-number risk.

6. Build-time template leakage guard
- Status: completed
- Implemented in: manuscript/Makefile
- Notes: combined build now fails if unresolved Jinja tokens remain.

7. External benchmark source hardening
- Status: completed
- Implemented in: scripts/analyze_consistency_trends.py
- Notes: default benchmark source changed to local artifacts to avoid unintended remote drift in standard runs.

8. Build and evidence verification
- Status: completed
- Evidence: targeted pytest subset passes; manuscript render and journal-check pass; figures target executes under updated workflow.
