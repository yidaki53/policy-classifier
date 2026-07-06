---
section_id: "04_significance"
section_title: "Conclusion"
objective: "Explain why the results matter, what is reproducibly established, and what limitations or next checks remain."
required_inputs:
  - "manuscript/sections/03_results.md"
  - "Latest consistency/recency/SARIMAX summaries under output/analysis/."
required_outputs:
  - "Concise significance interpretation tied to current evidence."
  - "Explicit caveats tied to linkage coverage, windows, and model assumptions."
required_metrics:
  - "Reference to key summary metrics from results where relevant."
required_figures_tables:
  - "none (may reference figures produced in results section)"
provenance_requirements:
  - "Claims must only generalize beyond what current reproducible metrics support."
update_triggers:
  - "Any material change in core findings or uncertainty bounds."
  - "Any major methodology revision."
owner: "manuscript-agent"
status: "active"
last_updated_utc: "2026-07-02T23:25:00Z"
---

# Conclusion

This section closes the manuscript's central story. We began with a societal and academic gap: difficulty evaluating whether parliamentary conduct matches party claims at scale. We then built and tested a reproducible multimodal estimator that remains inspectable end to end. The significance is that this workflow turns fragmented parliamentary signals into a single auditable measurement process with explicit uncertainty boundaries.

In practical terms, the manuscript contributes more than party-level scores. It also makes those scores inspectable from origin to interpretation. Readers can see where information enters the pipeline, how signals are transformed, where uncertainty is introduced, and which assumptions materially affect comparative conclusions. This transparency is essential for policy-relevant computational work, where it often matters as much as raw predictive performance.

The deterministic chain matters because it keeps the analysis traceable from raw parquet inputs through linkage, classification, annual summaries, recency weighting, and SARIMAX time-series modeling. This makes party comparisons and election-runup checks reproducible rather than purely model-dependent. It also enables sensitivity checks for linkage coverage, window definitions, and seasonal specification. In this framing, the core contribution is a behavior-based operational metric of party ideology under stated assumptions, not recovery of a single externally defined ground-truth scale.

This distinction should be explicit. A behavior-based metric asks, "what pattern is visible in observed parliamentary records under declared assumptions?" It does not ask, "what is the true ideology of a party in a metaphysical sense?" By preserving that distinction, the manuscript avoids overclaiming while still offering a useful comparative instrument for political analysis. This interpretative framing aligns with recent cross-national work on measuring party positions from parliamentary debate [@ebrecht2024cross] and with multimodal approaches that combine text, speech, and voting records [@jaursch2025multimodal].

The results provide descriptive evidence in the direction of all three working hypotheses: H1 (modality-sensitive profiles) is supported by the observed motion/speech/action divergence patterns; H2 (systematic variation in say-do consistency) is supported by the cross-party consistency score distribution; H3 (variation in fulfillment and contradiction diagnostics) is supported by the spread in party-level fulfillment summaries and the consistency-versus-fulfillment comparison. The refreshed held-out speech evaluation accuracy is `0.2033`, while isotonic recalibration raises top-1 accuracy to `0.3709` on the same set; temperature scaling leaves top-1 accuracy unchanged and should be treated as a calibration-only transform. All interpretations carry the uncertainty qualifications described below.

External benchmark validation remains a triangulation check. The current benchmark summary reports overlap `n=8`, Spearman `0.2857` with bootstrap CI in `output/analysis/party_ideology_benchmark_validation.json`. These comparisons are not used as definitive ground truth for the behavior-based ideology metric.

In the current refresh, structural-vs-all stability still shows measurable drift (`abs max delta ≈ 0.152` in `output/analysis/link_strata_stability_summary.json`), so outputs should still be interpreted as comparative diagnostics under explicit modeling assumptions rather than as a fully validated single latent-trait estimate.

These caveats are not a weakness of the study design. They are a methodological safeguard. Parliamentary language and action records are complex social data, and explicit uncertainty treatment is necessary to keep claims scientifically proportional to evidence quality. The manuscript therefore treats uncertainty reporting as part of the contribution, not merely as a limitations paragraph.

**Note on classifier accuracy context**: The baseline speech accuracy of `0.2033` is against a `7`-class problem where random chance gives approximately `0.143`. This metric is currently evaluated against Britannica-based category definitions (label ontology), not an external latent-ideology ground truth. The observed value is ~1.4x chance, indicating meaningful structure in the signal but substantial residual uncertainty. All downstream modality-level comparisons should therefore be interpreted as exploratory estimates with calibrated probabilities rather than validated class assignments.

Taken together, the manuscript supports a bounded claim. Multimodal parliamentary evidence can produce auditable, updateable, and policy-relevant ideology estimates when each modeling choice is justified and each interpretation is tied to reproducible artifacts.

The broader implication is procedural as well as substantive. Procedurally, the workflow provides a reusable template for future legislatures and periods: ingest, classify, link, calibrate, stratify, and report with full provenance. Substantively, it supports accountable public reasoning by making it easier to compare what parties advocate with what enters institutional action pathways, while preserving explicit limits on causal interpretation.

The immediate quid ergo is practical. For public accountability, the framework provides a transparent way to compare what parties claim with what they advance through parliamentary pathways. For journalistic and civil-society monitoring, it offers a reproducible update cycle that can flag widening gaps between rhetoric and action without implying causal intent. For comparative political analysis, it provides an auditable measurement protocol that can be rerun across legislative periods, rather than a one-off index that cannot be stress-tested.

## Future Research Directions

Short-horizon work should focus on measurement reliability. Priority items include expanded labeled speech evaluation, tighter calibration diagnostics by party/topic strata, and additional linkage-ablation tests that quantify which conclusions are most sensitive to fallback pathways.

Medium-horizon work should test transferability across institutions. The deterministic-first architecture is portable, but category definitions, linkage assumptions, and calibration behavior should be re-estimated in legislatures with different party systems and procedural regimes before cross-country comparisons are interpreted.

Long-horizon work should couple this descriptive framework to stronger identification designs. The present analysis can motivate future causal designs on agenda effects, coalition bargaining, and policy uptake, but those questions require quasi-experimental variation and assumptions not claimed here.

The contribution is also ecosystem-facing. This workflow complements, rather than replaces, manifesto coding, expert-survey positioning, and vote-scaling traditions. Each captures a different slice of political behavior; combining them can improve triangulation, while preserving transparency about what each measure can and cannot establish.