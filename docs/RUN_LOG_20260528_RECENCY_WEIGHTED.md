# Run Log: Full-Corpus Recency-Weighted Analysis

## Metadata

- UTC timestamp: 2026-05-29
- Entry points:
  - `scripts/analyze_consistency_trends.py`
  - `scripts/analyze_recency_weighted_trends.py`
  - `scripts/build_speech_motion_linkage.py`
- Thermal settings:
  - `CLASSIFIER_CPU_FRACTION=0.15`
  - sleep cadence in full-corpus classifier defaults (`sleep_every=50`, `sleep_seconds=0.2`)

## Inputs (full corpus artifacts)

- `data/parquet/speech_classifications_with_rhetoric_full.parquet` (339500 rows)
- `data/parquet/classifications.parquet` (1420475 rows)
- `output/analysis/promise_fulfillment_party_topic_year.parquet` (403 rows)
- `output/analysis/party_ideology_drift_by_modality_year.parquet` (117 rows)

## Outputs

- `output/analysis/recency_weighted_party_scores.parquet`
- `output/analysis/recency_weighted_parliament_timeseries.parquet`
- `output/analysis/recency_weighted_summary.json`
- `output/analysis/consistency_score_party.parquet`
- `output/analysis/lead_lag_speech_to_action_party_year.parquet`
- `output/analysis/parliament_direction_over_time.parquet`

## Linkage check

- Speech `rel_dok_id` values bridge to `data/betankande/parquet/*.parquet` committee documents.
- Coverage summary on the current full corpus: 8,150 unique speech `rel_dok_id` values, 2,744 matching betänkande documents, and 2,531 of those committee documents reference at least one motion ID.
- `scripts/build_speech_motion_linkage.py` now prefers this committee bridge before falling back to direct motion IDs or proximity/category matching.

## Election-runup check

- `scripts/analyze_recency_weighted_trends.py` now reports a runup split using the existing election-year set.
- Current recency summary: parliament runup action index `2.9244` vs non-runup `2.9528`, for a difference of `-0.0284` on the current full-corpus aggregate.

## Notes

- This run uses full existing classification artifacts, not sample subsets.
- Recency weighting uses exponential decay with half-life = 3 years.
- Manuscript section `manuscript/sections/03_results.md` updated with reproducibility-oriented summary text.
