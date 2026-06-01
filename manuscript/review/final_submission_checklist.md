# Final Submission Checklist (2026-05-31)

## Build and compliance

- [x] `make render` passes.
- [x] `make combined` passes.
- [x] `make pdf` passes.
- [x] `make journal-check` passes (`status: ready`).
- [x] Final size recorded: 17 pages, 4263 words.
- [x] Post-remediation rerun completed: `make render`, `make combined`, and `make journal-check` all pass (2026-05-31).
- [x] Updated PDF rebuilt after second-tranche edits (`make pdf` completed, 2026-05-31).

## Readability and structure

- [x] First-mention definitions added for key metrics.
- [x] Strict sentence-length and passive-voice pass completed.
- [x] Punctuation micro-pass (comma/semicolon simplification) completed.
- [x] Intermediate/process figures moved to appendix.
- [x] Caption rendering fixed and visually verified in the rebuilt PDF.
- [x] Question framing expanded to state channel-specific ideology assumptions and transferability limits.
- [x] Results reorganized into auditable subsections (key evidence, coverage/model quality, cross-modality contrasts, robustness limits).
- [x] Conclusion strengthened with explicit quid-ergo audience framing and a structured future-research roadmap.

## Visual QA

- [x] Page-by-page visual inspection completed (pages 1-17).
- [x] Front-page title heading corrected (removed generic `Title` heading).
- [x] Appendix Figure 8 regenerated with excluded-party policy aligned to overlay rules (`Unknown`, `Moderaterna`, `Vänsterpartiet`, `X`).
- [x] Post-remediation visual QA rerun completed on updated PDF pages (through appendix and references).
- [x] Residual resolved: appendix figure legibility improved after source-level plot tightening (`scripts/analyze_consistency_trends.py`) and expanded renderer width overrides (`scripts/render_manuscript_jinja.py`) for benchmark, modality overlay, voting cohesion, three-way divergence, and speech-heatmap figures; post-fix PDF visual QA confirms acceptable readability.

## Reproducibility and provenance

- [x] Figure captions include update timestamps.
- [x] Results provenance block includes scripts, outputs, and UTC timestamp.
- [x] Journal requirements report generated at `manuscript/build/journal_requirements_report.json`.
- [x] Linkage summary count set to the full speech corpus (`n_speeches_with_category = n_speeches`) and validated in code.
- [x] Focused regressions added for speech-analysis vote coding, renderer exclusion policy, and manuscript rendering.
- [x] Outputs rebuilt and revalidated after the latest code changes.
- [x] Previously unresolved abstract/results/conclusion template fields replaced with concrete, artifact-traceable values in manuscript source sections.
- [x] Data Availability expanded with release-snapshot archival workflow guidance (tag + commit + persistent archive record).
- [x] Cross-section metric synchronization audit passed for key anchors across rendered abstract/results/conclusion (corpus counts, evaluation N, accuracy, and NLL metrics).
- [x] Count semantics clarified: manuscript now distinguishes speech-category rows (`n=991235`) from unique speeches (`n=141605`).
- [x] Linkage confidence composition clarified with all strata (graph-signatory, existing-reference, heuristic fallback, structural-high) and rounded reporting policy.

## Methods and interpretation hardening

- [x] Methodology now includes explicit design-tradeoff rationale for parquet-first execution, deterministic-first scoring, fairness-constrained linkage, and non-causal inference boundaries.
- [x] Benchmark interpretation boundaries clarified (CHES as triangulation signal, not ground-truth oracle).
- [x] Appendix now includes a compact limits-to-claims matrix mapping each evidence component to supported claims, non-supported claims, uncertainty source, and caution statement.
- [x] Methodology now explicitly discloses fulfillment-imputation handling (`--fulfillment-fill`) as a conditional assumption.
- [x] Sensitivity verification executed for fill settings (`0.0` vs `0.5`): current party-level consistency outputs are unchanged in this refresh (`rho=1.000`, max abs delta `0.000`).

## Submission readiness

- [x] Data Availability statement now includes concrete public repository access and release/commit citation instructions.
- [x] Figure 8 regenerated and visually verified under aligned excluded-party policy.

## Commit scope recommendation (manuscript-only)

Stage only these files for the manuscript readability/finalization patch:

- `manuscript/sections/01_title.md`
- `manuscript/sections/01a_abstract.md`
- `manuscript/sections/02_question.md`
- `manuscript/sections/03_methodology.md`
- `manuscript/sections/03_results.md`
- `manuscript/sections/04_significance.md`
- `manuscript/sections/07_appendix.md`
- `manuscript/sections/05_data_availability.md`
- `scripts/render_manuscript_jinja.py`
- `manuscript/review/reviewer2_critique.md`
- `manuscript/review/final_submission_checklist.md`
