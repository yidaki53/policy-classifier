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
last_updated_utc: "2026-05-29T00:00:00Z"
---

# Abstract

Most public debate about party ideology relies on what parties say in manifestos or campaign messaging. At the same time, citizens and researchers need tools that track whether parliamentary conduct aligns with those claims. Existing computational pipelines often emphasize one modality (text or votes). They also optimize predictive performance without preserving transparent audit trails across the full workflow. We address this gap with a reproducible, parquet-first framework that estimates party policy positioning from three connected channels: motions, speeches, and roll-call votes.

Our objective is to operationalize a behavior-based ideology metric grounded in observed parliamentary activity. In plain terms, we estimate where parties stand by looking at what they formally propose, how they publicly argue, and how they actually vote. The pipeline starts from a transparent deterministic rule baseline (fixed and inspectable decision rules). It then adds embedding, zero-shot, and transformer signals in an ensemble design (a weighted combination of model families). Finally, it aligns modalities through explicit linkage and fairness-constrained optimization.

On the current full corpus, the workflow covers `n=202925` motions (1971-2024), `n=991235` speech-category rows corresponding to `n=141605` unique speeches (2014-2026), and `n=21464` unique roll-call vote events (1993-2026). With full speech-action linkage in the final stage, party-level consistency outputs are exported as auditable parquet artifacts. In labeled speech evaluation (`n=2656`), baseline accuracy is `0.2033`; baseline NLL is `2.1535`, with calibration NLL `1.9221` (temperature) and `1.7115` (isotonic). Recency-weighted and lead-lag analyses provide party and parliament trajectories over time, and SARIMAX model selection is tracked through saved trial artifacts for reproducible forecasting diagnostics.

Linkage diagnostics (latest production refresh): full linkage coverage is achieved by design (`n=141605` linked rows), with confidence-stratified composition. In `output/analysis/speech_action_link_confidence_summary.json`, graph-signatory links are `n=95822` (`67.7%`), existing-reference links are `n=20841` (`14.7%`), heuristic fallback links are `n=12576` (`8.9%`), and structural high-confidence links are `n=12366` (`8.7%`). Action counts are near balanced (vote `n=71403`, motion `n=70202`).

Across the manuscript, the story runs from problem to evidence to implication. We define the measurement gap, implement an auditable multimodal estimator, test robustness under linkage and calibration uncertainty, and report descriptive party-level contrasts with explicit non-causal boundaries. The contribution is both methodological and empirical: a transparent parliamentary analysis stack that researchers can update, inspect, and stress-test as new data arrive. We use external references for triangulation (a directional comparison check), not as definitive ground truth, because statement-based benchmarks may diverge from realized parliamentary action.

The next section states the research question and comparative frame that govern these empirical claims.