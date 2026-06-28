---
_agent_frontmatter:
  id: analysis.log_viz_summary_20260628"
  purpose: "Summary of generated figures and pipeline log status for review-ready manuscript assessment"
  steward: analysis
  edit_policy: generated_do_not_edit
  generator: swedish_parliament_policy_classifier.codebase_review
---

## Pipeline log: 2026-06-28T16:11:16Z

Run finished `2026-06-28T17:00:05Z` with **12/12 steps OK** (figures: 11/12 at `17:59:24`; manuscript: `18:00:05`).

Top-level step health:
- `api_check` — checked `anforande`, `votering`, `mot`, `prop`, `bet`, `latest_dates`
- `download` — speeches, votering, betankande, bulk → rc=0
- `extract` — speeches, votering, betankande → rc=0
- `api_fetch` — motions, speeches
- `extract_new_sources` — questions, betankande, interpellations → rc=0
- `classify` — speeches (rc=0), motions (rc=0), rhetorical_adjustment (rc=0)
- `classify_new_sources` — questions, betankande, interpellations → rc=0
- `link_prop_bet`, `merge_prop_bet`
- `analysis` — linkage, profiles, link_all_speeches, axis_alignment, contradiction, contradiction_by_modality, link_confidence, uncertainty, consistency, link_stability, latent, recency → rc=0
- `figures` — manuscript_motion_figures, party_profiles, party_profiles_advanced, interactive, voting, speech_profiles, overlay (rc=0), three_way
- `manuscript` — render (rc=0), combined (rc=0)

**No failures or non-zero return codes.**

## Generated visualizations (manuscript figures)

The following 6 PNGs were produced in `output/manuscript/figures/` and referenced in `manuscript/sections/03_results.md`.

| Figure | File | What the result indicates |
|--------|------|---------------------------|
| **Consistency vs Fulfillment** | `figure_consistency_vs_fulfillment.png` | Scatter of parties by mean consistency (x) vs fulfillment (y). Upper-right parties are text-action coherent *and* push policy through to linked action. Lower-left parties diverge across modalities or have weaker pathway continuation. This is the primary descriptively grounded party-comparison plot. |
| **Parliament Direction Over Time** | `figure_parliament_direction_over_time.png` | Recency-weighted parliament-wide ideology trajectory (0=far left → 1=far right) with 95% bootstrap CIs. Shifts indicate aggregate directional movement in the parliamentary center of gravity; widening CIs flag periods of higher uncertainty or linkage volatility. |
| **Speech vs Action ideology (Quid Ergo)** | `figure_quid_ergo_speech_vs_action.png` | Per-party speech-side vs action-side ideological placement. Distance from the diagonal is the say–do gap: large separations mean rhetoric and final parliamentary action differ; near-diagonal parties exhibit cross-channel coherence. |
| **Pareto frontier: consistency vs vote fidelity** | `figure_pareto_frontier_consistency_fidelity.png` | Normalized trade-off space between consistency and vote fidelity. Parties near the upper-right frontier are high-coherence *and* high-fidelity in vote alignment; those near the lower-left face at least one weakness. |
| **Consistency vs Fulfillment by benchmark year** | `figure_consistency_fulfillment_vs_benchmark_party_year.png` | Party-year variation in consistency/fulfillment with benchmark overlays. Drift here shows whether apparent party differences are stable over time or period-specific. |
| **Modality overlay by party** | `figure_modality_overlay_by_party.png` | Per-party stacked profiles across speech/motion/vote channels. Channel divergence within a party motivates the multimodal measurement design and motivates figure 3 (say–do gap). |

**Interpretive import for the paper (all claims remain descriptive, not causal):**
- These visualizations support the core descriptive claim that *party ideology is not single-channel.* Motion-only and speech-only portraits can yield materially different rankings.
- The parliament-direction plot contextualizes year-to-year interpretation: an apparent shift in one party can be read as large-parameter movement instead of a uniquely idiosyncratic trajectory.
- The Pareto frontier figure delivers the headline trade-off framing: no party simultaneously maximizes both cross-channel consistency and vote fidelity at the same time under current linkage assumptions.
- All plots are grounded in the current `output/analysis_rhetorical/*.parquet` exports and `output/parquet/*.parquet` classification tables generated in this same run, so provenance is unambiguous.

### Other artifact directories that support these plots
- `output/analysis_rhetorical/consistency_score_party.parquet`
- `output/analysis_rhetorical/consistency_fulfillment_party_year.parquet`
- `output/analysis_rhetorical/ideological_gap_party.parquet`
- `output/analysis_rhetorical/parliament_direction_over_time.parquet`
- `output/analysis_rhetorical/promise_fulfillment_party_summary.parquet`
- `output/analysis_rhetorical/speech_action_axis_scores.parquet`