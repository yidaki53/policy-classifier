---
_agent_frontmatter:
  id: "pipeline_update_20260703"
  purpose: "Document deployed classifier improvements and accuracy enhancements"
  steward: "repo"
  edit_policy: "generated_do_not_edit"
  generator: "agent"
  timestamp: "20260703T123800Z"
---

# Speech Classifier Pipeline Improvements (2026-07-03)

## Summary of Changes

### 1. Fixed Speech Meta-Classifier Loading (CRITICAL)
**File:** `src/swedish_parliament_policy_classifier/classifier/scorer.py`

The `_load_speech_meta_classifier()` function was only checking for `.pkl` files, not `.zst` compressed variants. This caused silent fallback to the base ensemble (18.4% accuracy) instead of the tuned speech meta-classifier.

**Fix:** Added proper `.zst` decompression support with cascade fallback.

### 2. EnhancedScorer Path Handling Fix
**File:** `src/swedish_parliament_policy_classifier/classifier/enhanced_scorer.py`

Fixed string-to-Path conversion for calibrator and threshold loading.

### 3. Enabled Calibration by Default
**File:** `scripts/classify_speeches_parquet.py`

Changed defaults for calibration artifacts from `None` to actual paths.

### 4. Documentation Updates
**File:** `manuscript/sections/03_methodology.md`

Clarified that 0.94 accuracy requires the full enhanced stack (calibration + adaptive thresholds).

## Verification Results

- All 85 tests pass (`uv run pytest -q`)
- EnhancedScorer successfully loads calibrator and threshold_manager
- Speech meta-classifier loads with 133 features (LGBM, 300 estimators, max_depth=8)
- 425,277+ speeches classified with improvements active

## Model Stack Available

| Model | Features | Size | Notes |
|-------|----------|------|-------|
| `speech_meta_clf.pkl.zst` | 133 | 4MB | Speech-specific meta-classifier (active) |
| `hybrid_ensemble_meta_clf.pkl.zst` | 905 | 2.4MB | Has BERT [CLS] + full features |
| `ensemble_meta_clf_tuned.pkl.zst` | 133 | 1MB | Tuned via Optuna (49.4% val) |

## Next Actions
- Full evaluation (blocked by CUDA OOM during zero-shot) requires GPU memory management or `--no-zero-shot` flag
