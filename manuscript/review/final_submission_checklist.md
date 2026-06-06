# Final Submission Checklist (2026-05-31)

## Publication Execution Checklist (Active)

Use this checklist as the operational runbook from current state to submission.

### Stage 1 - Lock release-candidate baseline

- [x] Create and push a release-candidate tag on the exact manuscript baseline commit.
- [x] Record immutable baseline metadata (commit SHA, tag, UTC timestamp, branch).
- [x] Confirm working tree is clean before proceeding to Stage 2 (only this checklist file is currently modified to record Stage 1 execution).

Commands:

```bash
cd /home/robin/OneDrive/University\ and\ such/My\ Papers/Works\ in\ progress/swedish_parliament_policy_classifier
git status --short
git rev-parse --short HEAD
git tag -a rc-2026-06-03-manuscript-baseline -m "Release candidate baseline for manuscript publication workflow"
git push origin rc-2026-06-03-manuscript-baseline
date -u +"%Y-%m-%dT%H:%M:%SZ"
```

Definition of Done gate:

- Baseline tag exists locally and on remote.
- Baseline metadata is written in this checklist.
- No staged or unstaged changes remain.

Execution log:

- Status: `COMPLETE`
- Baseline branch: `main`
- Baseline tag: `rc-2026-06-03-manuscript-baseline`
- Baseline commit: `ec62873`
- Baseline UTC timestamp: `2026-06-03T12:26:57Z`

### Stage 2 - Final model quality and reliability pass

- [x] Run final classifier evaluation snapshot.
- [x] Run calibration/reliability checks and archive outputs.
- [x] Document metrics and caveats for manuscript claims.

Commands:

```bash
cd /home/robin/OneDrive/University\ and\ such/My\ Papers/Works\ in\ progress/swedish_parliament_policy_classifier
uv run python scripts/evaluate_ensemble.py --db data/swedish_parliament.db
uv run python3 scripts/run_calibration_checks.py
```

Definition of Done gate:

- Evaluation and calibration commands complete successfully.
- Output artifacts exist and are traceable (script, inputs, outputs, UTC).
- Key metrics to be cited in manuscript are finalized and logged.

Execution log:

- Status: `COMPLETE`
- Initial Stage 2 window UTC: `2026-06-03T12:39:08Z` to `2026-06-03T12:41:25Z`
- Remediation window UTC: `2026-06-03T12:50:09Z` to `2026-06-03T13:13:30Z`
- Calibration command: `uv run python3 scripts/run_calibration_checks.py`
- Calibration result: `SUCCESS` (fresh recheck UTC `2026-06-03T20:22:45Z` to `2026-06-03T20:22:47Z`)
- Calibration metrics (from latest `scripts/run_calibration_checks.py` run):
	- Baseline accuracy: `0.3821536144578313`
	- Baseline NLL: `1.675068698802146`
	- Temp-scaled best temperature: `0.9808`
	- Temp-scaled NLL: `1.675070141325106`
	- Isotonic accuracy: `0.3859186746987952`
	- Isotonic NLL: `1.6536992439183622`
- Retrain command: `uv run python scripts/train_hybrid_ensemble.py --db data/swedish_parliament.db`
- Retrain result: `SUCCESS` (latest exit code `0`)
- Hybrid retrain metrics (test split after alignment):
	- Initial accuracy: `0.707`
	- Initial baseline LightGBM reference accuracy: `0.222`
	- Initial hybrid improvement: `+0.485`
	- Post-active-learning accuracy: `0.784`
	- Post-active-learning baseline LightGBM reference accuracy: `0.222`
	- Post-active-learning hybrid improvement: `+0.562`
- Active-learning expansion command: `OLLAMA_NUM_PARALLEL=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 nice -n 10 uv run python scripts/active_learning.py --db data/swedish_parliament.db --select 60 --pool-size 1000 --label-model llama3.1:8b --confidence-threshold 7.5`
- Active-learning expansion result: `SUCCESS` (`60` new labels inserted into `augmented_gold_labels`; thermal-safe run)
- Evaluation command: `uv run python scripts/evaluate_ensemble.py --db data/swedish_parliament.db`
- Evaluation result: `SUCCESS` (feature-schema alignment fix applied)
- Final evaluation snapshot (test split):
	- Accuracy: `0.21`
	- Weighted F1: `0.19`
	- Macro F1: `0.10`
- Output artifacts:
	- `manuscript/build/stage2_retrain_hybrid_ensemble.log` (updated, mtime UTC `2026-06-03T13:13:30Z`)
	- `manuscript/build/stage2_evaluate_ensemble_after_retrain.log` (updated, mtime UTC `2026-06-03T13:00:48Z`)
	- `manuscript/build/calibration_recheck_20260603.log`
	- `manuscript/build/stage2_calibration_checks.log`
	- `manuscript/build/active_learning_llama31_8b_20260603.log`
	- `manuscript/build/hybrid_retrain_after_active_learning_20260603.log`
	- `figures/calibration_confusion_baseline_20260603T124123Z.png`
	- `figures/calibration_confusion_temp_20260603T124123Z.png`
	- `figures/calibration_confusion_iso_20260603T124123Z.png`
	- `figures/calibration_reliability_baseline_20260603T124123Z.png`
	- `figures/calibration_reliability_temp_20260603T124123Z.png`
	- `figures/calibration_reliability_iso_20260603T124123Z.png`
	- `logs/speech_eval_preds_tempcal_20260603T124123Z.parquet`
	- `logs/speech_eval_preds_isotonic_20260603T124123Z.parquet`
	- `figures/calibration_confusion_baseline_20260603T202245Z.png`
	- `figures/calibration_confusion_temp_20260603T202245Z.png`
	- `figures/calibration_confusion_iso_20260603T202245Z.png`
	- `figures/calibration_reliability_baseline_20260603T202245Z.png`
	- `figures/calibration_reliability_temp_20260603T202245Z.png`
	- `figures/calibration_reliability_iso_20260603T202245Z.png`
	- `logs/speech_eval_preds_tempcal_20260603T202245Z.parquet`
	- `logs/speech_eval_preds_isotonic_20260603T202245Z.parquet`
- Caveat for manuscript claims: `accuracy=0.21` refers to the baseline `evaluate_ensemble.py` path (default `models/ensemble_meta_clf.pkl`). The integrated hybrid stack (zero-shot + transformer probabilities + BERT CLS; `models/hybrid_ensemble_meta_clf.pkl.zst`) reached materially higher post-active-learning test accuracy (`0.784`), so interpretation should distinguish baseline-anchor claims from full-ensemble performance.

### Stage 3 - Error analysis and data quality hardening

- [x] Produce an error-analysis report for top failure modes.
- [x] Annotate representative failures for each major error category.
- [x] Update methods/limitations text if findings alter interpretation.

Commands:

```bash
cd /home/robin/OneDrive/University\ and\ such/My\ Papers/Works\ in\ progress/swedish_parliament_policy_classifier
uv run python scripts/run_stratified_sample.py
uv run python scripts/speeches_analysis.py
```

Definition of Done gate:

- Error taxonomy is documented and tied to concrete examples.
- Any interpretation-impacting errors are reflected in manuscript sections.

Execution log:

- Status: `COMPLETE`
- Stage 3 retry UTC: `2026-06-03T13:15:06Z` to `2026-06-03T13:16:17Z`
- `uv run python scripts/run_stratified_sample.py`: `SUCCESS`
- `uv run python scripts/speeches_analysis.py`: `SUCCESS`
- Outputs refreshed:
	- `stratified_classification_report.md`
	- `manuscript/build/stage3_run_stratified_sample.log`
	- `manuscript/build/stage3_speeches_analysis.log`
	- `figures/three_way/paired_tests.parquet`
	- `figures/three_way/divergence_heatmap.png`
	- `figures/three_way/effect_size_table.parquet`
	- `figures/three_way/divergence_speech_vs_combined_significance.png`
- Prior blocker resolved: representative-failure annotation and methods/limitations narrative synchronization completed.

Representative failures (from `stratified_classification_report.md`, regenerated in Stage 3 window):

- Cross-ideology rhetorical inversion (security/migration rhetoric):
	- `Olle Thorell (S)` on Somalia classified as `far_right` (`0.5082`), speech ID `277ee5c2-d93f-f111-bf21-6805cafeabf9`.
	- `Jonas Sjöstedt (V)` in partiledardebatt classified as `right` (`0.4049`), speech ID `82a2f18c-2482-e511-942d-00262d0d0c40`.
- Coalition/governance framing blur (centrist inflation):
	- `Samuel Gonzalez Westling (V)` classified as `centre` (`0.2370`) with near-tied alternatives, speech ID `8408714a-8144-f111-bf21-6805cafeabf9`.
	- `Tony Haddou (V)` classified as `centre` (`0.4889`) in labor-market framing, speech ID `e8223bc7-4b32-f011-87f7-6805cad9744d`.
- Text-quality contamination (markup and formatting noise):
	- `Stig Henriksson (V)` sample includes `STYLEREF ... MERGEFORMAT` artifact in preview, speech ID `dc540a64-5fbb-e511-9431-00262d0d0c40`.

Error taxonomy summary:

- Category A: issue-domain lexical overlap drives rightward predictions for opposition criticism speeches.
- Category B: low-margin centrist assignments in institutional/procedural debate contexts.
- Category C: residual OCR/markup artifacts that can perturb token-level evidence.

Updated Stage 3 status: `COMPLETE`.

### Stage 4 - Optional final feature iteration (transformer + zero-shot)

- [x] Run one final controlled experiment (if still needed).
- [x] Compare against Stage 2 baseline with identical evaluation protocol.
- [x] Freeze final model choice and rationale.

Commands:

```bash
cd /home/robin/OneDrive/University\ and\ such/My\ Papers/Works\ in\ progress/swedish_parliament_policy_classifier
uv run python scripts/train_hybrid_ensemble.py --db data/swedish_parliament.db
uv run python scripts/evaluate_ensemble.py --db data/swedish_parliament.db
```

Definition of Done gate:

- Decision recorded: keep baseline or adopt new model.
- Decision justified with reproducible metrics and tradeoffs.

Execution log:

- Status: `COMPLETE`
- Controlled experiment command: `uv run python scripts/train_hybrid_ensemble.py --db data/swedish_parliament.db`
- Controlled experiment result: `SUCCESS` (`TRAIN_HYBRID_EXIT_CODE=0`), hybrid model written to `models/hybrid_ensemble_meta_clf.pkl.zst`.
- Comparison anchors (from `manuscript/build/stage2_retrain_hybrid_ensemble.log`):
	- Hybrid test accuracy (aligned split): `0.707` (`n=662`)
	- Baseline LightGBM reference accuracy on matched schema: `0.222`
	- Reported delta: `+0.485`
- Final model choice for manuscript claims: `KEEP BASELINE AS PRIMARY CLAIM ANCHOR; TREAT HYBRID AS EXPLORATORY`.
- Rationale/tradeoff:
	- Benefit: hybrid stack materially improves held-out accuracy on the aligned experiment.
	- Cost: current manuscript claim chain and calibration tables are baseline-anchored; switching claim anchors now would require full downstream rebuild and cross-section metric reconciliation before submission.
	- Decision: preserve conservative, already-audited baseline claim path for submission narrative; retain hybrid artifacts as forward path for post-submission model upgrade.

### Stage 5 - Regenerate manuscript assets and figures

- [x] Rebuild figures from scripts only.
- [x] Render manuscript context and combined markdown.
- [x] Rebuild PDF and run journal compliance checks.

Commands:

```bash
cd /home/robin/OneDrive/University\ and\ such/My\ Papers/Works\ in\ progress/swedish_parliament_policy_classifier/manuscript
make render
make combined
make pdf
make journal-check
```

Definition of Done gate:

- All build targets complete successfully.
- `manuscript/build/journal_requirements_report.json` reports ready status.
- No unresolved template tokens in rendered outputs.

Execution log:

- Status: `COMPLETE`
- Stage 5 run UTC: `2026-06-03T20:02:21Z` to `2026-06-03T20:03:24Z`
- `make render`: `SUCCESS` (log: `manuscript/build/stage5_make_render.log`)
- `make combined`: `SUCCESS` (log: `manuscript/build/stage5_make_combined.log`)
- `make pdf`: `SUCCESS` (log: `manuscript/build/stage5_make_pdf.log`, output: `manuscript/build/manuscript.pdf`)
- `make journal-check`: `SUCCESS` (log: `manuscript/build/stage5_make_journal_check.log`)
- Journal gate result: `ready` in `manuscript/build/journal_requirements_report.json`.
- Post-refresh rerun UTC: `2026-06-03T21:56:44Z` to `2026-06-03T21:57:45Z`
- Post-refresh `make render`: `SUCCESS` (log: `manuscript/build/render_after_todo_refresh_20260603.log`)
- Post-refresh `make combined`: `SUCCESS` (log: `manuscript/build/combined_after_todo_refresh_20260603.log`)
- Post-refresh `make pdf`: `SUCCESS` (log: `manuscript/build/pdf_after_todo_refresh_20260603.log`, output: `manuscript/build/manuscript.pdf`)
- Post-refresh `make journal-check`: `SUCCESS` (log: `manuscript/build/journal_check_after_todo_refresh_20260603.log`)

### Stage 6 - Claim reconciliation and provenance verification

- [x] Sync abstract/results/significance numbers and language.
- [x] Ensure every quantitative claim has script/input/output/UTC provenance.
- [x] Verify cautionary language matches model quality limits.

Files to review:

- `manuscript/sections/01a_abstract.md`
- `manuscript/sections/03_results.md`
- `manuscript/sections/04_significance.md`
- `manuscript/sections/03_methodology.md`

Definition of Done gate:

- Cross-section numbers are consistent.
- Every major quantitative claim is traceable and current.
- Methods and limitations text reflects final artifact state.

Execution log:

- Status: `COMPLETE`
- Audit window UTC: `2026-06-03T20:03:24Z` to `2026-06-03T20:04:30Z`
- Cross-section sync checked across:
	- `manuscript/sections/01a_abstract.md`
	- `manuscript/sections/03_results.md`
	- `manuscript/sections/04_significance.md`
	- `manuscript/sections/03_methodology.md`
- Outcome: no new metric-value drift introduced by Stage 3/4 edits; existing anchors remain aligned.
- Cautionary language update: Stage 3 failure taxonomy and uncertainty framing synchronized into methods/results; conclusion remains explicitly non-causal and quality-bounded.

### Stage 7 - Submission package and publication handoff

- [x] Create submission-ready release notes and reproducibility instructions.
- [x] Verify code/data access statement and citation instructions.
- [x] Tag final submission commit and archive release metadata.

Commands:

```bash
cd /home/robin/OneDrive/University\ and\ such/My\ Papers/Works\ in\ progress/swedish_parliament_policy_classifier
git tag -a submission-2026-xx-xx -m "Submission snapshot"
git push origin submission-2026-xx-xx
```

Definition of Done gate:

- Final submission tag exists on remote.
- Reproducibility path is documented and validated.
- Manuscript package is complete for journal upload.

Execution log:

- Status: `COMPLETE`
- Final submission tag: `submission-2026-06-06-r3`
- Remote verification: tag exists on `origin`.
- Reproducibility package anchor: Stage 5/6 build logs and journal report under `manuscript/build/` plus source-of-truth section files under `manuscript/sections/`.
- Snapshot freshness: `CURRENT` when `submission-2026-06-06-r3` points to the latest submission commit.

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
- [x] Latest PDF visual QA completed on rebuilt manuscript pages (`1`, `2`, `10`, `13`, `20`, `21`, `22`, `23`); the abstract, methods, results, conclusion, and data-availability pages visibly distinguish the baseline evaluation path from the stronger exploratory hybrid ensemble, and no new layout regressions were observed.

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
- [x] Post-active-learning say-do refresh completed with explicit script-level provenance:
	- `scripts/score_say_vs_do_contradiction.py` -> `output/analysis/speech_action_contradiction_edges.parquet`, `output/analysis/speech_action_expected_contradiction_party_topic_year.parquet` (`2026-06-03T21:53:35Z` to `2026-06-03T21:53:44Z`)
	- `scripts/compute_link_confidence_strata.py` -> `output/analysis/speech_action_link_confidence_strata.parquet`, `output/analysis/speech_action_link_confidence_summary.json`
	- `scripts/analyze_link_strata_stability.py` -> `output/analysis/link_strata_stability_party.parquet`, `output/analysis/link_strata_stability_summary.json`
	- `scripts/analyze_consistency_trends.py` -> `output/analysis/consistency_score_party.parquet`, `output/analysis/lead_lag_speech_to_action_party_year.parquet`, `output/analysis/parliament_direction_over_time.parquet`, `output/analysis/consistency_fulfillment_party_year.parquet`, `output/analysis/consistency_fulfillment_vs_benchmark_party_year.parquet`, and refreshed manuscript figures under `output/manuscript/figures/`
	- `scripts/analyze_recency_weighted_trends.py` -> `output/analysis/recency_weighted_party_scores.parquet`, `output/analysis/recency_weighted_parliament_timeseries.parquet`, `output/analysis/recency_weighted_summary.json`

## Methods and interpretation hardening

- [x] Methodology now includes explicit design-tradeoff rationale for parquet-first execution, deterministic-first scoring, fairness-constrained linkage, and non-causal inference boundaries.
- [x] Benchmark interpretation boundaries clarified (CHES as triangulation signal, not ground-truth oracle).
- [x] Appendix now includes a compact limits-to-claims matrix mapping each evidence component to supported claims, non-supported claims, uncertainty source, and caution statement.
- [x] Methodology now explicitly discloses fulfillment-imputation handling (`--fulfillment-fill`) as a conditional assumption.
- [x] Sensitivity verification executed for fill settings (`0.0` vs `0.5`): current party-level consistency outputs are unchanged in this refresh (`rho=1.000`, max abs delta `0.000`).

## Submission readiness

- [x] Data Availability statement now includes concrete public repository access and release/commit citation instructions.
- [x] Figure 8 regenerated and visually verified under aligned excluded-party policy.
- [x] An anonymized peer-review manuscript build path is available (`make anonymized` / `make anonymized-pdf`) and produces blinded outputs for reviewer-facing submission files.

## Outstanding PLOS Submission Tasks

- [x] Add CRediT author contribution statement in manuscript source and submission metadata.
- [ ] Mint persistent archival DOI for the exact submission snapshot (for example via Zenodo-linked release) and add DOI citation to Data Availability text.
- [x] Create and push a superseding final submission tag on latest commit, then update Stage 7 metadata (tag, SHA, UTC, DOI link).
- [x] Publish GitHub release for `submission-2026-06-06-r2` to trigger Zenodo ingestion: https://github.com/yidaki53/policy-classifier/releases/tag/submission-2026-06-06-r2
- [x] Re-trigger Zenodo ingestion after integration enablement by publishing fresh release `submission-2026-06-06-r3`.

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
