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
last_updated_utc: "2026-06-29T12:00:00Z"
---

# Methodology

We focus on Sweden because the parliamentary record offers unusually high institutional traceability for this research objective [@carlson2024swedish]. We can observe party-level behavior consistently across motions, plenary speeches, and roll-call voting, all tied to a transparent legislative process. This setting reduces ambiguity about where parties make claims and where they record actions. That clarity is necessary because our core goal is to compare political speech with parliamentary conduct, not to maximize cross-country breadth.

We use official Riksdag open data as the primary source because each modality contributes a distinct inferential role that no single source can replace. Motions capture formal policy proposals. Speeches capture rhetorical framing and agenda emphasis. Votes capture enacted parliamentary choices under institutional constraints. The choice to combine these sources is not cosmetic. We need this combination to test whether observed ideology depends on what we measure as statement, proposal, or action, following recent multimodal approaches in political analysis [@jaursch2025multimodal; @osnabrugge2023speech].

## Measurement architecture

The modeling strategy is deterministic-first and multimodal by design [@barbera2021automated]. The core architecture can be summarized as "rules for reliability, models for flexibility." We first use fixed, inspectable rules as a baseline — keyword matching against ideological category definitions and regex patterns — because they provide stable behavior across reruns and can be checked line by line. We then add three learned components as controlled extensions:

- **Embedding similarity** computes semantic distance between speech text and each category's precomputed embedding, catching paraphrases that keyword matching misses [@nikolaev2023multilingual; @miok2022multiaspect].
- **Zero-shot NLI entailment** tests whether the text supports or criticizes each category's position, providing the strongest protection against rhetorical inversion — where a speaker quotes an opponent's language but opposes their position [@alvarez2021label; @patz2025german].
- **A fine-tuned Swedish BERT classifier** provides domain-specific probability vectors trained on motion-level gold labels [@devlin2019bert; @wolf2020transformers].

These five signal types (keyword + regex + embedding + zero-shot + BERT) are combined in a LightGBM meta-learner trained on 2,656 gold-label speeches, achieving 0.94 per-category accuracy on held-out test data. A rhetorical pattern detection layer, derived from Britannica-based keyword lists, multiplicatively boosts categories whose ideological framing signals are detected. The meta-learner receives both raw and rhetorically-adjusted probabilities, learning optimal weighting for the speech domain.

This hybrid design trades simplicity for better representational capacity while retaining a transparent baseline. Each component was added only after stratified validation showed a specific failure mode that existing components could not address: embedding scores catch semantic similarity but generate false positives when a speaker discusses a topic without endorsing it; zero-shot NLI corrects these false positives via the critique hypothesis term; BERT provides calibration for formal policy language where embeddings are noisy.

## Linkage, aggregation, and inference boundaries

Explicit linkage joins speech outputs with vote-side records so speech claims and parliamentary actions can be compared directly [@proksch2015politics]. We use fairness-constrained linkage [@grech2025fairness] to reduce coverage artifacts: if some parties or time periods are easier to link than others, apparent ideological differences can become artifacts of missingness rather than meaningful political change [@mikhaylov2012catch]. The linkage threshold is tuned via a fairness-constrained genetic algorithm that balances coverage across parties and time windows.

After linkage, the pipeline aggregates aligned outputs into party-year and party-topic-year diagnostics: ideology index (the party-level left-right position implied by category-weighted outputs), consistency (cross-modality agreement), fulfillment (the share of speech-linked pathways that proceed to vote-side action records), and contradiction (mismatch intensity between speech-side and action-side positions). These follow established scaling approaches for legislative text [@lowe2011scaling; @lauderdale2016scaling] and pledge-fulfillment methodology [@carlson2024swedish].

We distinguish three model-quality references. The speech-specific meta-classifier is the primary anchor for speech-level claims, trained on speech gold labels. The motion-trained baseline, when transferred to speeches without adaptation, achieves only 0.20 accuracy — demonstrating that motion and speech linguistic registers differ materially. The integrated hybrid stack (which adds zero-shot and transformer probability features on top of BERT and base features) serves as a cross-model sensitivity check.

Benchmark alignment with external sources such as the Chapel Hill Expert Survey (CHES) is used for directional triangulation only, because expert-survey ideology captures perception and positioning at a different abstraction level than behavior observed in parliamentary records [@ebrecht2024cross]. Agreement with CHES strengthens confidence that party ordering is not arbitrary; disagreement can be substantively informative when speech and vote behavior diverge from expert-position estimates under coalition or agenda constraints.

We treat all outputs as conditional on this design envelope. Calibration and sensitivity checks are built in at each stage. We test whether model confidence is aligned with observed correctness on labeled data, and we run sensitivity checks to see whether headline conclusions remain stable when linkage thresholds and weighting settings are varied. All numerical and statistical computations build on the scikit-learn [@pedregosa2011scikit], pandas [@mckinney2010data], and NumPy [@harris2020array] ecosystem. With these design choices justified, the next section reports the empirical patterns.

## Reproducibility

The full pipeline implementation is available in the project repository. The complete analysis can be reproduced with a single command:

```bash
uv run python scripts/update_pipeline.py --cpu-fraction 0.25
```

The speech classification pipeline supports resume-by-skip for incremental processing. Linkage, fairness controls, and temporal diagnostics are executable scripts that export auditable parquet artifacts — each encoding version strings that identify which classifier components were active, enabling downstream filtering by component subset. All scripts are executed in a pinned Python environment using `uv` (see `pyproject.toml` for exact dependency versions).
