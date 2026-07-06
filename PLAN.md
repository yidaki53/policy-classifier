# Implementation Plan

## Phase 1: Fix ideological placement (critical — correctness)
1. Change `_ideology_score()` to compute left-right balance excluding centre as neutral
2. Clean up centre category keywords (remove governance/administration terms)
3. Update all downstream scripts that use the old formula

## Phase 2: Filter defunct parties
1. Create a canonical `CURRENT_PARTIES` constant
2. Apply filter in all visualization scripts

## Phase 3: Create per-party trend visualizations
1. Create `scripts/generate_party_trends.py` with:
   - Party ideology trendlines (last 15 years)
   - Party fulfillment trendlines (last 15 years)

## Phase 4: Fix bibliography
1. Integrate orphaned references into appropriate sections

## Phase 5: Restructure manuscript narrative
1. Move detailed pipeline description to appendix
2. Remove file paths from prose
3. Strengthen narrative arc

## Phase 6: Fix margin overflow
1. Add LaTeX margin settings
2. Reduce inline code verbosity
3. Fix appendix figure widths