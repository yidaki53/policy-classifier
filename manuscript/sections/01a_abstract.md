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

{{ abstract_metrics_paragraph }}

We interpret outputs as descriptive diagnostics under explicit non-causal boundaries. The contribution is a transparent, auditable measurement stack that can be updated and stress-tested as new parliamentary data arrive. The next section states the research question and comparative frame.