---
section_id: "03_methodology"
section_title: "Methodology"
objective: "Document and justify data choices, model choices, and inference boundaries for a reproducible multimodal measurement pipeline."
required_inputs:
  - "Current parquet artifacts for motions, speeches, and voting"
  - "Current classification, linkage, and aggregation scripts"
required_outputs:
  - "A methods narrative that explains why each major choice was made and what tradeoff it addresses"
required_metrics:
  - "Key uncertainty and robustness diagnostics used for non-causal interpretation"
required_figures_tables:
  - "Method coverage and modality-alignment diagnostics (where applicable)"
provenance_requirements:
  - "All described methods must map to executable scripts and artifacts"
update_triggers:
  - "Any change in data source, linkage logic, model family, or calibration settings"
owner: "manuscript-agent"
status: "active"
last_updated_utc: "2026-05-30T00:00:00Z"
---

# Methodology

We focus on Sweden because the parliamentary record offers unusually high institutional traceability for this research objective. We can observe party-level behavior consistently across motions, plenary speeches, and roll-call voting, all tied to a transparent legislative process. This setting reduces ambiguity about where parties make claims and where they record actions. That clarity is necessary because our core goal is to compare political speech with parliamentary conduct, not to maximize cross-country breadth.

We use official Riksdag open data as the primary source because each modality contributes a distinct inferential role that no single source can replace. Motions capture formal policy proposals. Speeches capture rhetorical framing and agenda emphasis. Votes capture enacted parliamentary choices under institutional constraints. The choice to combine these sources is not cosmetic. We need this combination to test whether observed ideology depends on what we measure as statement, proposal, or action.

The modeling strategy is deterministic-first and multimodal by design. We keep deterministic rules as the baseline because they provide inspectable decision logic and stable behavior across reruns. That stability is essential for auditability in an academic setting. We then add embedding, zero-shot, and transformer-based components as controlled extensions. These components improve coverage of linguistic variation and contextual nuance where fixed rules are too brittle. This layered design trades some simplicity for better representational capacity while retaining a transparent baseline for comparison.

To keep this design auditable, we treat major methodological choices as explicit tradeoffs rather than hidden defaults. We choose parquet-first processing over database-centric reporting workflows to preserve reproducible snapshot semantics and faster columnar analysis. We choose deterministic rules as the first layer to maximize inspectability, then add learned components to reduce linguistic brittleness. We choose fairness-constrained linkage over unconstrained matching to reduce party/time coverage artifacts, even when this increases engineering complexity. We choose descriptive inference boundaries over causal language because the current evidence structure is observational and linkage-sensitive.

The core architecture can be summarized as "rules for reliability, models for flexibility." In practical terms, we first use fixed rules that can be checked line by line, and then add learned models to catch meaning that appears in varied wording. We choose this hybrid setup instead of a single black-box model because the goal is to measure and explain party behavior, not only to maximize prediction scores. The tradeoff is straightforward: the system is more complex to build, but easier to audit and better at handling language variation.

For interpretive consistency, we define the core reported constructs once and apply them uniformly. Ideology index denotes the party-level location implied by category-weighted modality outputs (motions, speeches, votes). Consistency denotes cross-modality agreement after alignment and aggregation. We report it as a descriptive index, not a causal parameter. Fulfillment denotes the observed share of speech-linked policy pathways that proceed to vote-side action under current linkage assumptions. Contradiction denotes modeled mismatch intensity between speech-side positions and action-side outcomes. We use it as a diagnostic component of the broader consistency/fulfillment interpretation, not as a standalone causal estimate.

Because these terms can be confused, we keep their roles separate. Ideology index answers "where does this party appear to stand, based on observed behavior?" Consistency answers "how similar are the signals across speech, motions, and action records?" Fulfillment answers "of what is argued in speech, how much continues into linked action pathways?" Contradiction answers "how strongly do speech-side and action-side signals point in different directions?" None of these metrics is a moral grade or a causal estimate. Each is a structured summary of observed behavior under explicit assumptions.

The ensemble and alignment flow follows a fixed sequence. First, deterministic, embedding, zero-shot, and transformer components score motions. Next, a meta-learner combines those signals. Then explicit linkage joins speech outputs with vote-side records. Finally, the pipeline aggregates aligned outputs into party-year and party-topic-year diagnostics for consistency and fulfillment.

Calibration and uncertainty checks are built in at each stage because model confidence is easy to over-interpret. In this study, a confidence score means how sure the model is internally. It does not guarantee factual correctness. We therefore test whether confidence is aligned with observed correctness on labeled data. We also run sensitivity checks to see whether headline conclusions remain similar when linkage and weighting settings are changed.

We use fairness-constrained linkage and calibration diagnostics because speech and vote evidence are not uniformly available across parties, years, and topics. Without these controls, apparent ideological shifts can become artifacts of missingness or linkage imbalance rather than meaningful political change. We therefore report descriptive estimates with uncertainty diagnostics and avoid causal claims.

The fairness-constrained linkage step is especially important for interpretation. Linkage means connecting speeches to later parliamentary action records so speech and action can be compared directly. If some parties or time periods are easier to link than others, a simple comparison can look different just because the data are uneven. Rebalancing and confidence-level breakdowns are therefore used as bias controls. They do not remove all uncertainty, but they lower the risk that apparent party differences are mainly linkage artifacts.

Missing-pathway handling is also explicit. In the consistency workflow (`scripts/analyze_consistency_trends.py`), the fulfillment component is configurable through `--fulfillment-fill` and is used when a party-topic observation has no available motion pathway after linkage. This is a fairness tradeoff rather than a hidden default: imputing a neutral or low value avoids mechanically penalizing parties with sparse linkage support, but it also introduces an assumption that must be tested. We therefore treat fulfillment-imputation sensitivity as part of model uncertainty and report results as descriptive estimates conditional on the selected setting. In the latest refresh, no party-level missing-motion pathways remained after linkage aggregation, so fill sensitivity (`0.0` vs `0.5`) produced identical party-level consistency outputs; this is reported as a verification result rather than assumed ex ante.

We selected the reference set for function rather than citation breadth. Foundational text-as-data and political scaling studies define the measurement assumptions used here. Parliamentary NLP work motivates modality-specific choices in debate text processing and interpretation. We use external benchmarks such as CHES for triangulation only, because expert-survey ideology captures perception and positioning at a different abstraction level than behavior observed in parliamentary records.

Benchmark choice also carries an interpretation tradeoff. External references such as the Chapel Hill Expert Survey (CHES) improve directional plausibility checks, but they do not provide a direct ground-truth target for parliamentary behavior trajectories in this pipeline. Agreement with CHES can strengthen confidence that party ordering is not arbitrary; disagreement can be substantively informative when speech and vote behavior diverge from expert-position estimates under coalition or agenda constraints. We therefore treat benchmark alignment as a validity signal, not a correctness oracle.

The executable implementation maps directly to this rationale. Deterministic-first scoring and hybrid aggregation are defined in `scripts/classify.py` and `src/swedish_parliament_policy_classifier/classifier/scorer.py`. Linkage and fairness controls are implemented in linkage and consistency-tuning scripts under `scripts/`. Recency and temporal diagnostics are generated by `scripts/analyze_recency_weighted_trends.py` and SARIMAX artifacts exported under `output/analysis/`. This script-to-artifact mapping is the operational mechanism that keeps methods claims auditable.

Finally, we treat all outputs as conditional on this design envelope. The method supports reproducible descriptive inference about parliamentary behavior, not universal claims about "true" ideology. Result quality depends on dictionary design, category definitions, linkage quality, and classifier calibration. The artifact chain explicitly documents and tests each of these components.

With these design choices justified, the next section reports the empirical patterns and concrete examples that follow from this measurement framework.