---
_agent_frontmatter:
  id: "reviewer_guide"
  purpose: "Step-by-step reproducibility guide for reviewers of the Swedish Parliament Policy Classifier manuscript"
  steward: "manuscript-agent"
  edit_policy: "manual"
  generated: "2026-06-24T19:30:00Z"
---

# Reviewer Guide: Reproducibility and Verification

This guide helps a reviewer verify that the manuscript's claims are reproducible from the provided code and data artifacts.

## 1. Quick Environment Setup

```bash
# Clone the repository
git clone https://github.com/yidaki53/policy-classifier.git
cd policy-classifier

# Create and activate the managed environment
uv venv create .venv
source .venv/bin/activate

# Install dependencies (pinned via uv.lock)
uv pip install -e .

# Optional but recommended for full pipeline:
uv pip add spacy sentence-transformers
uv run python3 -m spacy download sv_core_news_sm
```

**Python version:** 3.12 (managed by `uv`)
**Key dependencies:** pandas, pyarrow, scikit-learn, transformers, torch, lightgbm, spacy, sentence-transformers, huggingface_hub

## 2. Full Reproduction Command

The single entry point for the entire analysis pipeline is:

```bash
make incremental-update
```

This runs `scripts/update_pipeline.py --cpu-fraction 0.25`, which executes, in order:

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `scripts/download_speeches.py` | Download new Riksdag speech ZIPs |
| 2 | `scripts/download_votering.py` | Download new vote CSV ZIPs |
| 3 | `scripts/download_betankande.py` | Download new committee report ZIPs |
| 4 | `scripts/import_bulk_datasets.py` | Import ZIPs to parquet |
| 5 | `scripts/extract_speeches.py` | Extract speech parquet shards |
| 6 | `scripts/extract_votering.py` | Extract vote parquet |
| 7 | `scripts/classify_speeches_parquet.py` | **Classify speeches** (resume-safe) |
| 8 | `scripts/classify.py --raw data/parquet/raw_motions.parquet` | Classify motions |
| 9 | `scripts/apply_rhetorical_adjustments.py` | Apply rhetorical multipliers |
| 10 | `scripts/build_speech_motion_linkage.py` | Link speeches to motions |
| 11 | `scripts/build_profiles.py` | Aggregate party profiles |
| 12 | `scripts/analyze_consistency_trends.py` | Consistency/fulfillment |
| 13 | `scripts/analyze_promise_fulfillment.py` | Promise-fulfillment |
| 14 | `scripts/analyze_ideological_gap.py` | Ideological gap |
| 15 | `scripts/analyze_recency_weighted_trends.py` | Recency-weighted scores |
| 16 | `scripts/generate_figures.py` | Manuscript figures |

Thermal-safe defaults: CPU fraction 0.25, sleep-every 50 speeches, sleep 0.2s. Override with `CLASSIFIER_CPU_FRACTION=0.5` for faster runs on cooled hardware.

## 3. Key Artifacts and Their Provenance

### Data inputs (parquet)
```
data/speeches/parquet/*.parquet       # Raw speech shards (33 files covering 1994-2026)
data/votering/parquet/*.parquet       # Roll-call vote records
data/betankande/parquet/*.parquet     # Committee reports
data/parquet/raw_motions.parquet      # Normalized motions
```

### Classification outputs
```
data/parquet/speech_classifications.parquet   # Long-form: one row per (speech, category)
                                               # Columns: speech_id, category, raw_score,
                                               # normalized_weight, all_category_probs_json,
                                               # matched_rules, classifier_version, confidence
data/parquet/classifications.parquet          # Motion classifications (long-form)
```

### Analysis outputs (under `output/analysis_rhetorical/`)
```
party_ideology_drift_by_modality_year.parquet   # Party x modality x year ideology scores
consistency_score_party.parquet                 # Cross-modality consistency by party
promise_fulfillment_party_summary.parquet      # Fulfillment rates by party
speech_action_link_confidence_strata.parquet   # Linkage confidence breakdown
recency_weighted_party_scores.parquet          # Recency-weighted scores
party_latent_ideology_estimates.parquet        # Latent ideology estimates
```

### Figures
```
figures/manuscript/                  # Publication-ready PNG/PDF figures
manuscript/build/rendered_sections/  # Rendered Markdown for manuscript
```

## 4. How to Verify a Specific Claim

### "n=152,062 unique speeches classified"
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('data/parquet/speech_classifications.parquet', columns=['speech_id'])
print(f'Unique speeches: {df[\"speech_id\"].nunique()}')
print(f'Total rows: {len(df)}')
"
```

### "0.94 accuracy on held-out speech gold labels"
```bash
uv run python scripts/evaluate_speech_gold_labels.py
# Inspect logs/speech_eval_preds_*.parquet for per-category accuracy
```

### "Linkage: n=141,605 linked rows, 67.7% graph-signatory"
```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('output/analysis_rhetorical/speech_action_link_confidence_strata.parquet')
print(df.groupby('link_source').size())
print(f'Total: {len(df)}')
"
```

## 5. Resume and Crash Safety

The pipeline is designed to survive interruption:

- **Speech classification**: Skips speech_ids already present in the output parquet. If a speech hangs (>300s), it is logged to `logs/classify_hang.log` and skipped on next run.
- **Crash diagnostics**: SIGSEGV/SIGABRT handlers write to `logs/classify_crash.log` with speech_id and phase.
- **Thermal safe**: Respects `CLASSIFIER_CPU_FRACTION` and sleeps every N speeches.

##4. Bibliography

All 26 references are cited in `02_question.md` and `03_methodology.md`:
```
manuscript/bibliography/references.bib
```

## 5. Common Reviewer Questions

| Question | Answer |
|----------|--------|
| "Can I run this myself?" | Yes — `make incremental-update` from repo root. |
| "How long does classification take?" | ~30-60 minutes on a laptop CPU for full corpus (152k speeches). Resume-safe, so partial runs can continue. |
| "What if a speech hangs?" | It times out after 300s, is logged, and skipped on resume. Check `logs/classify_hang.log`. |
| "What hardware is needed?" | ~8GB RAM recommended. CPU-only is fine. GPU optional for embedding/transformer layers. |
| "Is the data public?" | Yes — Swedish Riksdag open data (CC0). Raw inputs are downloaded from `data.riksdagen.se`. |
| "Where are the exact claim-to-artifact mappings?" | Each section file's `provenance_requirements` field lists required inputs/outputs. The `stratified_classification_report.md` documents validation results. |
| "How do I check the PDF builds?" | `cd manuscript && make pdf` (requires pandoc + LaTeX). |
| "What tests should I run?" | `uv run pytest -q` — 53 tests pass. |

## 6. What to Skip

- `data/swedish_parliament.db` — legacy SQLite, no longer used in the parquet-first pipeline.
- `scripts/classify.py` — legacy motion classifier; use `scripts/classify_speeches_parquet.py` for speeches.
- `web/` — annotation UI, not needed for manuscript reproduction.