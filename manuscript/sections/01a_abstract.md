---
section_id: "01a_abstract"
section_title: "Abstract"
objective: "Provide a concise structured summary aligned with target-journal constraints and current reproducible outputs."
required_inputs:
  - "Current full-chain results from manuscript/sections/03_results.md"
  - "Current methods scope across motions, speeches, and votes"
required_outputs:
  - "A <=300-word abstract that states objective, methods, key results, and conclusion"
required_metrics:
  - "At least one anchor metric with n and/or key performance indicators"
required_figures_tables:
  - "none"
provenance_requirements:
  - "All quantitative claims must be traceable to current analysis artifacts"
update_triggers:
  - "Any material result refresh"
  - "Any major methods change"
owner: "manuscript-agent"
status: "active"
last_updated_utc: "2026-06-08T22:40:42Z"
---

# Abstract

Most computational studies of party ideology rely on manifesto positions or single-modality text. We present a reproducible, parquet-first framework that estimates Swedish party policy profiles from three parliamentary channels: motions, speeches, and roll-call votes. The pipeline uses deterministic rules as an inspectable baseline, then adds embedding, zero-shot, and transformer signals in an ensemble, with explicit linkage and fairness controls across modalities.

On the current full corpus, the workflow covers `n=202925` motions (1971-2024), `n=141605` unique speeches (2014-2026), and `n=21464` vote events (1993-2026). The speech-specific meta-classifier achieves `0.94` per-category accuracy on held-out speech gold labels (`n=2656`). The motion baseline transferred to speeches yields only `0.2033` accuracy, demonstrating register-transfer difficulty; the integrated hybrid ensemble achieves `0.784` post-active-learning test accuracy. Full speech-action linkage (`n=141605` rows) is stratified by confidence: `67.7%` graph-signatory, `14.7%` existing-reference, `8.9%` heuristic fallback, `8.7%` structural high-confidence.

Consistency between speech and vote-linked action varies substantially across parties: SD shows the highest aggregate consistency (0.565) while C shows the lowest (0.518), and fulfillment rates differ by a factor of two across parties. We interpret outputs as descriptive diagnostics under explicit non-causal boundaries. The contribution is a transparent, auditable measurement stack that can be updated and stress-tested as new parliamentary data arrive.
