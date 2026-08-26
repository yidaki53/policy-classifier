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
last_updated_utc: "2026-07-02T23:30:00Z"
---

# Abstract

Most computational studies of party ideology rely on manifesto positions or single-modality text. We present a reproducible, parquet-first framework that estimates Swedish party policy profiles from three parliamentary channels: motions, speeches, and roll-call votes. The pipeline uses deterministic rules as an inspectable baseline, then adds embedding, zero-shot, and transformer signals in an ensemble, with explicit linkage and fairness controls across modalities.

On the current full corpus, the workflow covers `n=202926` motions (2007-2026), `n=425276` speeches (1993-2026), and `n=21464` unique roll-call vote events (1993-2026). With full speech-action linkage in the final stage, party-level consistency outputs are exported as auditable parquet artifacts. In labeled speech evaluation (`n=2656`), baseline accuracy is `0.2033`; baseline NLL is `2.1535`, with calibration NLL `1.9221` (temperature) and `1.7115` (isotonic). Recency-weighted and lead-lag analyses provide party and parliament trajectories over time, and SARIMAX model selection is tracked through saved trial artifacts for reproducible forecasting diagnostics.

We interpret outputs as descriptive diagnostics under explicit non-causal boundaries. The contribution is a transparent, auditable measurement stack that can be updated and stress-tested as new parliamentary data arrive. The next section states the research question and comparative frame.