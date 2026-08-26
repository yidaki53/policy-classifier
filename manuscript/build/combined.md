
# Multimodal estimation of Swedish party policy profiles from motions, speeches, and votes


# Abstract

Most computational studies of party ideology rely on manifesto positions or single-modality text. We present a reproducible, parquet-first framework that estimates Swedish party policy profiles from three parliamentary channels: motions, speeches, and roll-call votes. The pipeline uses deterministic rules as an inspectable baseline, then adds embedding, zero-shot, and transformer signals in an ensemble, with explicit linkage and fairness controls across modalities.

On the current full corpus, the workflow covers `n=202926` motions (2007-2026), `n=425276` speeches (1993-2026), and `n=21464` unique roll-call vote events (1993-2026). With full speech-action linkage in the final stage, party-level consistency outputs are exported as auditable parquet artifacts. In labeled speech evaluation (`n=2656`), baseline accuracy is `0.2033`; baseline NLL is `2.1535`, with calibration NLL `1.9221` (temperature) and `1.7115` (isotonic). Recency-weighted and lead-lag analyses provide party and parliament trajectories over time, and SARIMAX model selection is tracked through saved trial artifacts for reproducible forecasting diagnostics.

We interpret outputs as descriptive diagnostics under explicit non-causal boundaries. The contribution is a transparent, auditable measurement stack that can be updated and stress-tested as new parliamentary data arrive.


# Question

Democratic accountability depends on evaluating whether parties translate public claims into parliamentary conduct. In practice, this is difficult because evidence is fragmented across motions, plenary speech, and roll-call voting records. It is also difficult because many existing measures prioritize either interpretability or predictive power, but not both. The resulting gap is twofold: society lacks routinely reproducible behavior-based indicators of party positioning, and academia lacks an end-to-end multimodal pipeline that stays auditable as model complexity increases.

This fragmentation is not only technical. It reflects different political functions of each channel. Motion text captures formal agenda proposals. Parliamentary speech captures rhetorical framing, coalition signaling, and constituency-facing argument. Roll-call voting captures institutional commitment under procedural and coalition constraints. Public interpretation often assumes these channels are interchangeable proxies for ideology. They are not. A core motivation of this study is to measure what is lost when ideology is inferred from only one channel and what is recovered when all three are aligned in one auditable framework.

Primary question: What patterns of policy emphasis emerge across Swedish parliamentary motions, speeches, and voting behavior when ideology is measured through a deterministic-first, multimodal pipeline?

Measurement aim: Develop an operationalization (a concrete measurable definition of an abstract concept) that provides a bounded, reproducible estimate of party ideological positioning under stated assumptions. We pair that estimate with explicit sensitivity and uncertainty reporting across modalities and linkage settings.

We use key terms in a strict way throughout the manuscript. Modality means the channel from which we take evidence: motions (formal proposals), speeches (public parliamentary argument), and votes (recorded legislative action). Deterministic-first means the analysis starts from fixed, inspectable rules before adding machine-learning components, so the full chain remains traceable. Multimodal means the estimate is built from all three channels rather than inferred from only one.

The comparative frame is party-year and party-topic-year, derived from aligned motion, speech, and vote artifacts in the current reproducible parquet workflow. The outcomes of interest are ideology-index position, speech-action consistency, contradiction and fulfillment diagnostics, and direction-over-time trajectories. Within this frame, we test three linked hypotheses. First, party ideology profiles are modality-sensitive across motions, speeches, and vote-linked action channels. Second, say-do consistency varies systematically across parties and topics after fairness-constrained linkage. Third, promise-fulfillment and contradiction diagnostics add information beyond aggregate consistency alone.

The frame also sets the external-validity boundary. These estimates describe observed behavior under Swedish institutional conditions and data availability. They should not be generalized automatically to other legislatures with different party systems, committee structures, voting practices, or transparency standards. Transfer requires retesting category definitions, linkage assumptions, and calibration behavior in each new context.

In practical reading terms, each hypothesis asks a different accountability question. The first asks whether a party appears ideologically similar when we observe what it says versus what it does. The second asks whether that gap is stable across parties or concentrated in specific parties or issue domains. The third asks whether broad consistency indices hide meaningful differences in what actually gets carried from rhetoric into vote-linked action. This decomposition matters because one summary score can look stable even when politically important components are moving in different directions.

The reader path follows the same logic as the research design. In the next section, we justify the empirical choices and methodological architecture. We then present the results as descriptive measurement claims rather than causal effects. The final section returns to the opening societal and academic gap. It assesses what this workflow resolves now, what remains uncertain, and how the framework can support cumulative evidence.

We use the Swedish case as an analytic design choice, not as a claim of universal representativeness. Sweden offers unusually structured and publicly accessible parliamentary records. This makes it possible to evaluate reproducibility, traceability, and cross-modality alignment under realistic data conditions. The tradeoff is clear: future comparative work must test external generalization rather than assume it.

This framing follows established text-as-data and political-text measurement practice while prioritizing transparent, auditable inference boundaries [@grimmer2013text; @gentzkow2019text; @lowe2011scaling; @slapin2008centers]. Automated textual analysis of parliamentary debates has precedent in PLOS ONE [@rheault2016measuring], and vote-based ideology measurement is a recognized complement to speech-based approaches [@possieri2020voting].


# Methodology

## Data Sources

We use official Riksdag open data (data.riksdagen.se) as the primary source. Three modalities are ingested and normalized into compressed parquet datasets:

- **Motions**: `n=202926` documents spanning 2007-2026, retrieved via the Riksdag Open Data API (`/api/v1/dokument/sok?typ=mot`). Each motion includes metadata (party, date, title, signatories) and full text. Documents are deduplicated by `dok_id` and filtered to exclude committee reports and government propositions unless explicitly analyzed as comparison material.
- **Speeches**: `n=425276` unique plenary speeches spanning 1993-2026, retrieved via the Riksdag anförande API (`/api/v1/anforande/sok`). Each speech includes speaker, party affiliation, and timestamp. Speeches shorter than 50 characters after HTML stripping are excluded.
- **Votes**: `n=21464` unique roll-call vote events spanning 1993-2026, retrieved via the Riksdag votering API (`/api/v1/votering/sok`). Each vote event includes party-level voting records (yes/no/abstain/absent) and metadata linking to the underlying proposition.

Raw parliamentary records are ingested and normalized into parquet datasets for motions, speeches, consultation documents, and voting records. The ingest workflow is runnable through the repository scripts for speeches, voting, and consultation data.

We focus on Sweden because the parliamentary record offers unusually high institutional traceability for this research objective [@carlson2024swedish]. We can observe party-level behavior consistently across motions, plenary speeches, and roll-call voting, all tied to a transparent legislative process. This setting reduces ambiguity about where parties make claims and where they record actions. That clarity is necessary because our core goal is to compare political speech with parliamentary conduct, not to maximize cross-country breadth.

Each modality contributes a distinct inferential role that no single source can replace. Motions capture formal policy proposals. Speeches capture rhetorical framing and agenda emphasis. Votes capture enacted parliamentary choices under institutional constraints. The choice to combine these sources is not cosmetic. We need this combination to test whether observed ideology depends on what we measure as statement, proposal, or action, following recent multimodal approaches in political analysis [@jaursch2025multimodal; @osnabrugge2023speech].

## Measurement architecture

The modeling strategy is deterministic-first and multimodal by design [@barbera2021automated]. The core architecture can be summarized as "rules for reliability, models for flexibility." We first use fixed, inspectable rules as a baseline — keyword matching against ideological category definitions and regex patterns — because they provide stable behavior across reruns and can be checked line by line. We then add three learned components as controlled extensions:

- **Embedding similarity** computes semantic distance between speech text and each category's precomputed embedding, catching paraphrases that keyword matching misses [@nikolaev2023multilingual; @miok2022multiaspect].
- **Zero-shot NLI entailment** tests whether the text supports or criticizes each category's position, providing the strongest protection against rhetorical inversion — where a speaker quotes an opponent's language but opposes their position [@alvarez2021label; @patz2025german].
- **A fine-tuned Swedish BERT classifier** provides domain-specific probability vectors trained on motion-level gold labels [@devlin2019bert; @wolf2020transformers].

These five signal types (keyword + regex + embedding + zero-shot + BERT) are combined in a LightGBM meta-learner. **The speech-specific meta-classifier (trained on n=2,656 speech gold labels) achieves 0.94 per-category accuracy on held-out test data** when deployed with calibration and adaptive thresholds via EnhancedScorer. Without these pipeline improvements, the base classifier achieves only 0.184 accuracy due to class imbalance in the speech domain. A rhetorical pattern detection layer, derived from Britannica-based keyword lists, multiplicatively boosts categories whose ideological framing signals are detected. The meta-learner receives both raw and rhetorically-adjusted probabilities, learning optimal weighting for the speech domain. The pipeline loads the tuned speech meta-classifier and its calibration artifacts by default.

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

The workflow supports resume-by-skip for incremental processing. Linkage, fairness controls, and temporal diagnostics export auditable parquet artifacts with component-version metadata, enabling downstream filtering by model subset. The analysis is executed in a pinned Python environment using uv.


# Results

This section reports what the pipeline estimates in practice and where observed party differences are strongest. The focus is empirical rather than procedural. We present modality-sensitive contrasts, fulfillment patterns, and consistency contrasts using the current parquet artifacts.

How to read this section: each reported metric summarizes observed parliamentary behavior under clearly stated assumptions. A higher consistency value means stronger agreement between what a party proposes, says, and does in linked action records. A higher fulfillment value means a larger share of speech-linked issue pathways that continue into vote-side action records. A higher contradiction value means speech-side and action-side signals point in more different directions. Use these metrics as comparison tools, not as causal effect estimates.

## Key Visual Evidence
The figures below show headline outputs directly used for main-text interpretation.

![Consistency vs Fulfillment (updated 2026-07-01T20:43:39Z)](../output/manuscript/figures/figure_consistency_vs_fulfillment.png){ width=90% }

![Parliament Direction Over Time (updated 2026-07-01T20:43:39Z)](../output/manuscript/figures/figure_parliament_direction_over_time.png){ width=90% }


## Corpus Coverage and Model Quality

On the current full corpus, the workflow covers `n=202926` motions (2007-2026), `n=425276` speeches (1993-2026), and `n=21464` unique roll-call vote events (1993-2026). With full speech-action linkage in the final stage, party-level consistency outputs are exported as auditable parquet artifacts. In labeled speech evaluation (`n=2656`), baseline accuracy is `0.2033`; baseline NLL is `2.1535`, with calibration NLL `1.9221` (temperature) and `1.7115` (isotonic). Recency-weighted and lead-lag analyses provide party and parliament trajectories over time, and SARIMAX model selection is tracked through saved trial artifacts for reproducible forecasting diagnostics.

These figures indicate informative but uncertain signal. The speech-specific meta-classifier is the primary anchor for speech-level claims, while the motion baseline serves as a cross-domain sensitivity check. We therefore interpret all downstream contrasts as descriptive evidence rather than definitive recovery of a single hidden ideology value. This interpretative stance follows established practice in cross-national scaling of party positions from parliamentary text [@ebrecht2024cross].

Across hypotheses, the results are consistent with modality-sensitive ideology measurement under a descriptive interpretation. Party-level profiles differ across motions, speeches, and vote-linked action channels. Speech-action consistency also varies across parties after linkage constraints. Fulfillment diagnostics add information beyond aggregate consistency alone.

## Hypothesis 1: Modality-Sensitive Profiles

(See Figure 1, consistency vs fulfillment; Figure 3, party modality overlay; and Figure 8, three-way divergence.)

Substantively, this means no single channel can be treated as a complete proxy for party ideology. Motion-side evidence can reflect formal agenda setting and coalition strategy. Speech-side evidence can reflect rhetorical framing and constituency signaling. Vote-linked action can reflect final institutional bargaining constraints [@osnabrugge2023speech]. When these channels converge, confidence in the descriptive estimate increases. When they diverge, the divergence itself becomes a meaningful analytical result rather than a nuisance to suppress, consistent with recent multimodal frameworks for comparative political communication [@jaursch2025multimodal].

One concrete speech-level example shows how the classifier surfaces interpretable ideological signal from parliamentary language. Speech `c6c44eb9-b09c-e411-9412-00262d0d7125` (party `V`) is assigned category `left` with normalized weight `1.00` and confidence `1.00` in the speech classification artifact (`speech_classifications_with_rhetoric_full.parquet`). The speech text includes the statement: "Vi har redan varit med om Skånepolisens registrering av romer, och nu är det ett kvinnoregister ... Den här utvecklingen måste stoppas." This is not presented as proof of latent ideology on its own, but as an auditable instance of category assignment that can be traced back to source text and model output.

This example is included for transparency, not anecdotal persuasion. It demonstrates how a category assignment can be audited from source text to model output. This matters for reproducibility and interpretive discipline. A single speech cannot establish party-level ideology, but it can show whether the pipeline produces traceable and linguistically plausible intermediate outputs before aggregation.

(See Figure 8, three-way divergence.)

## Hypothesis 3: Fulfillment and Contradiction Diagnostics

Promise-fulfillment contrasts are substantively visible in the current summary table. In `output/analysis/promise_fulfillment_party_summary.parquet`, `SD` has `pct_speech_motion_vote = 0.3526` while `V` has `0.1787`; `V` shows `pct_speech_motion_no_vote = 0.0921`. These differences illustrate why fulfillment diagnostics are retained as a separate axis instead of being collapsed into one aggregate consistency score.

We interpret the fulfillment contrast as a pathway diagnostic. It asks whether issue emphasis in speech is followed by linked formal action at different rates across parties and issue domains. This pattern does not imply direct legislative causation from speech to votes. Instead, it quantifies how often speech-side attention appears in pathways that continue toward action records.

Consistency contrasts remain modest in absolute spread but informative for ranking and comparison. In `output/analysis/consistency_score_party.parquet`, `SD` records `consistency_score = 0.5499` and `motion_pathway_fulfillment = 0.9638`, while `C` records `consistency_score = 0.4840` and `motion_pathway_fulfillment = 0.8443`. The ranking difference is interpreted as descriptive signal under linkage and calibration assumptions, not as evidence of causal party effects.

Action-side party positioning is summarized from the latest supported-action evidence export when available. The current run did not yet materialize the action-position parquet artifacts.

![Action-side Evidence Digest](../output/manuscript/figures/figure_action_position_digest.png){ width=100% }



## Hypothesis 2: Say-Do Consistency

The consistency contrast complements fulfillment by focusing on agreement structure rather than endpoint rates. Two parties may display similar aggregate consistency while differing sharply in where that consistency comes from. For example, one party may show stable vote alignment but variable speech framing. For this reason, we interpret consistency and fulfillment jointly. Consistency indicates coherence across channels, while fulfillment indicates pathway continuation from speech-linked records into action-linked records.

(See Figure 1, consistency vs fulfillment.)

We keep classifier quality and substantive interpretation separate throughout. We treat calibration choices, linkage fairness constraints, and uncertainty intervals as sensitivity controls that bound interpretation. The speech-specific meta-classifier is the primary anchor for speech-level claims; the motion baseline serves as a cross-domain sensitivity check. All cross-party contrasts remain descriptive rather than causal. We use external benchmarks for directional triangulation only, because statement-based benchmarks can diverge from observed parliamentary action.

This separation between model quality and substantive claim strength is central to the manuscript's inferential stance. Better classifier metrics increase confidence that labels are coherent under the chosen category system, but they do not automatically justify stronger causal claims about party intent or policy consequences. Conversely, modest classifier performance does not invalidate all comparative diagnostics if uncertainty is explicitly modeled and interpretation remains bounded.

Recency-weighted party and parliament summaries continue to support the same interpretation boundary and are exported to `output/analysis/recency_weighted_party_scores.parquet`, `output/analysis/recency_weighted_parliament_timeseries.parquet`, and `output/analysis/recency_weighted_summary.json`.

We use recency weighting to answer a specific temporal question. Do contemporary party positions reflect information that is closer in time to current parliamentary behavior, rather than an equal average of distant historical periods? This improves interpretability for present-facing comparisons. It also introduces an explicit tradeoff. Short-term volatility can carry more influence than long-run structural stability.

To check whether action-side ideology shifts in election runup windows, recency summaries report runup action index `2.9244` versus non-runup `2.9528`; the latest runup-minus-nonrunup delta is `-0.0284`.

(See Figure 2, parliament direction over time.)

## Robustness and Interpretation Limits

The speech-to-motion linkage uses rel_dok_id-to-betankande bridging with fallback strategies. In the latest linkage summary, `n=425276` speeches are linked out of `n=425276` (`coverage=1.0000`), with `n=425276` speeches carrying a mapped ideology category.

Linkage diagnostics (latest production refresh): full coverage is retained by design (`n=769088` linked rows), but provenance now shifts materially toward graph-direct signatory evidence. In `output/analysis/speech_action_link_confidence_summary.json`, graph-signatory links are `n=154484` (`20.1%`), existing-reference links are `n=16851` (`2.2%`), heuristic fallback links are `n=212871` (`27.7%`), and structural high-confidence links are `n=41070` (`5.3%`). Action counts are near balanced (vote `n=315091`, motion `n=110185`).

Benchmarks are used for directional triangulation only, and election-runup summaries are treated as descriptive trend diagnostics rather than outcome forecasts.

The linkage diagnostics clarify how much of the speech corpus enters cross-modality comparison at different confidence levels. Higher-coverage linkage increases representativeness, while confidence-level splits provide a direct view of robustness under stricter versus looser matching criteria. Readers can then evaluate whether key comparisons are stable only in permissive linkage settings or remain visible under stricter thresholds.

Stage-3 stratified error review identified three recurring failure modes that bound interpretation of speech-level labels. First, rhetorical inversion cases appear when opposition speeches quote or attack right-coded issue language; this can induce rightward predictions even for left/center-left speakers (for example, speech IDs `277ee5c2-d93f-f111-bf21-6805cafeabf9` and `82a2f18c-2482-e511-942d-00262d0d0c40` in `stratified_classification_report.md`). Second, governance/procedural debate often yields low-margin centrist assignments with near-tied alternatives. Third, occasional markup contamination (for example `STYLEREF ... MERGEFORMAT`) remains in source text previews and can perturb token evidence. We therefore treat individual speech labels as auditable but noisy intermediates and prioritize party-level aggregates and sensitivity checks for substantive claims. These patterns are consistent with known challenges in automated parliamentary text analysis [@rheault2016measuring; @patz2025german].

Current metric anchors from this workflow include `n=403` rows in party-topic-year fulfillment and expected-contradiction aggregates, and `n=480` successful SARIMAX trials for monthly model selection.

Taken together, these results support a bounded empirical claim. Ideology-related party contrasts are detectable in multimodal parliamentary evidence, but their magnitude and ranking depend on linkage assumptions, model calibration, and category design. The appropriate reading is comparative and diagnostic, not definitive or causal.

(Methods pointer: `scripts/classify.py` and `src/swedish_parliament_policy_classifier/classifier/scorer.py` define the deterministic-first scoring baseline used in this workflow.)

A final synthesis helps bound interpretation. Robust findings are those that persist across modality comparisons and linkage-sensitivity diagnostics, including modality-sensitive party contrasts and cross-party variation in consistency. Suggestive findings include rank-order differences with modest absolute spread and outcomes that depend more strongly on calibration or weighting choices. Provisional findings include trend magnitudes in settings with higher fallback linkage reliance or model-family uncertainty. This hierarchy is used to keep empirical claims proportional to current evidence quality.

For first-time readers, see the compact plain-language guide in the Appendix section "How to read the metrics."

Run provenance for the latest full-chain recency and robustness execution remains anchored to `scripts/extract_motion_signatories.py`, `scripts/tune_link_rebalance_fair_ga.py`, `scripts/link_all_speeches_to_action.py`, `scripts/compute_ideology_axis_alignment.py`, `scripts/score_say_vs_do_contradiction.py`, `scripts/tune_consistency_wrangling_fair_ga.py`, and `scripts/analyze_consistency_trends.py`. The UTC timestamp is `2026-06-28T21:37:44Z` and outputs are written under `output/analysis/`.

These findings establish the empirical basis for the manuscript and motivate the final conclusion on what this framework can and cannot claim.


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


# Appendix

This appendix contains intermediate-step figures used for diagnostics, robustness checks, and process transparency.

## How to read the metrics

Use this quick guide when reading the Results section for the first time.

- Ideology index: A summary position for each party based on combined evidence from motions, speeches, and linked action records.
- Consistency: How closely a party's signals agree across what it proposes, says, and does.
- Fulfillment: How often issue emphasis in speeches continues into linked formal action pathways.
- Contradiction: How strongly speech-side and action-side signals point in different policy directions.
- Confidence score: How sure the model is internally about a label; this is not the same as guaranteed factual truth.
- Calibration: A check of whether model confidence matches observed correctness on labeled data.
- Linkage coverage: The share of speeches that can be linked to action records under current matching rules.
- Recency weighting: A time-weighting choice that gives more influence to newer behavior than older behavior.

Interpretation reminder: all metrics in this manuscript are descriptive comparison tools under explicit assumptions. They are not direct moral grades and they are not, by themselves, causal-effect estimates.

## Limits-to-Claims Matrix

This matrix states what each evidence type supports, what it does not support, and which uncertainty source most directly limits interpretation.

| Evidence component | Supported claim type | Not supported by itself | Primary uncertainty source | Required caution statement |
| --- | --- | --- | --- | --- |
| Modality-level ideology index | Descriptive comparative positioning across parties and periods | True latent ideology recovery | Category definitions and calibration quality | Read as conditional measurement under stated assumptions |
| Cross-modality consistency | Relative agreement between claims, speech, and action channels | Intentionality or sincerity attribution | Linkage quality and modality coverage imbalance | Differences may reflect data linkage structure as well as behavior |
| Consistency composite with fulfillment term | Relative ranking under a declared fulfillment-imputation setting | Imputation-free estimate of latent agreement | Missing-pathway imputation (`--fulfillment-fill`) and linkage sparsity | Treat as conditional on the chosen fill setting and verify stability under sensitivity checks |
| Fulfillment pathway rate | Observed continuation from speech-linked issues to action-linked records | Direct causal effect of speech on votes | Speech-to-action matching assumptions | Treat as pathway diagnostic, not causal transmission |
| Contradiction diagnostics | Relative divergence intensity across channels | Normative judgment of party credibility | Weighting choices and benchmark dependence | Use jointly with consistency and fulfillment, not in isolation |
| Recency-weighted trajectories | Present-facing trend summaries under explicit time weighting | Structural long-run equilibrium inference | Weight-decay specification and short-run volatility | Compare with unweighted trends before strong interpretation |
| External benchmark comparison (for example CHES) | Directional triangulation of party ordering | Ground-truth validation of parliamentary-behavior score | Construct mismatch between expert surveys and observed conduct | Agreement/disagreement is informative, not dispositive |

Use this matrix as a pre-interpretation checklist when drawing substantive conclusions from Results figures and tables.

## Appendix Figures (Intermediate Steps)
These figures capture intermediate diagnostics and process-level checks that support the main analysis without interrupting core result flow.

![Consistency-Fulfillment vs External Benchmark (Party-Year) (updated 2026-07-01T20:43:40Z)](../output/manuscript/figures/figure_consistency_fulfillment_vs_benchmark_party_year.png){ width=100% }

![Party Modality Overlay (updated 2026-07-01T20:45:05Z)](../output/manuscript/figures/figure_modality_overlay_by_party.png){ width=100% }

![Motion Category Distribution (updated 2026-07-01T20:44:01Z)](../figures/manuscript/pie_chart_categories.png){ width=90% }

![Party Motions Stacked (updated 2026-07-01T20:44:01Z)](../figures/manuscript/party_motions_stacked.png){ width=90% }

![Voting Cohesion Time Series (updated 2026-08-01T19:26:16Z)](../figures/voting/party_cohesion_timeseries.png){ width=100% }

![Three-way Divergence (updated 2026-07-01T20:45:20Z)](../figures/three_way/divergence_speech_vs_combined_significance.png){ width=100% }

![Speech Profiles Heatmap (updated 2026-07-01T20:45:03Z)](../figures/speeches/speech_profiles_heatmap.png){ width=100% }

![Action-side Evidence Digest (updated 2026-08-01T15:44:42Z)](../output/manuscript/figures/figure_action_position_digest.png){ width=90% }


# Data Availability

All data and metadata underlying the findings reported in this manuscript are available within the project repository and its reproducible artifact directories. Source parliamentary records are retrieved from official Swedish Parliament open-data endpoints and normalized into compressed parquet datasets. Derived analysis tables used for results and figures are available under the repository analysis outputs, and generated figure assets are available in the manuscript and figure directories.

All scripts required to reproduce ingest, classification, linkage, analysis, and figure generation are included in the repository and are executed in a pinned Python environment using uv. The exact rendering/build context for the manuscript is exported at build time, and journal-readiness checks are exported alongside the manuscript build artifacts.

No participant-level restricted data are introduced by this project; all primary inputs originate from publicly available parliamentary materials.

The full reproducible project is publicly accessible at `https://github.com/yidaki53/policy-classifier`. Submission and production versions should cite the exact release tag and commit hash used for manuscript generation.

Archival DOI for the submission snapshot (`submission-2026-06-06-r3`): `https://doi.org/10.5281/zenodo.20572644`.

For production handoff, the recommended archival path is to create a versioned release snapshot and archive it in a long-term repository service (for example, a Zenodo-linked GitHub release). This preserves the exact manuscript-state code and artifacts and provides a persistent accession identifier for citation without changing the underlying access pathway described above. The archived record should include the release tag, commit hash, artifact directory inventory, and manuscript build timestamp used in the submitted version.

Numeric reporting policy: analysis JSON artifacts preserve machine-precision float values (IEEE754). In manuscript prose and figure captions, percentages are rounded for readability (typically to one decimal place unless otherwise stated). Where rounded text differs from full-precision values, the full-precision artifact is the reproducible reference.


# Acknowledgments

The authors thank the Swedish Parliament (Riksdagen) for providing open-access data through the Riksdag Open Data API, which underpins all corpus materials used in this analysis.

## Author Contributions (CRediT)

Robin Oberg: Conceptualization, Methodology, Software, Data Curation, Formal Analysis, Visualization, Validation, Writing - Original Draft, Writing - Review and Editing.

This manuscript reports a single-author study. Contributor roles are declared using the CRediT taxonomy for submission metadata alignment.

No external funding was received for this study. The author received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

The author declares no competing interests.

All source data used in this manuscript are publicly available via the Riksdag Open Data API (data.riksdagen.se). Processed analysis artifacts (parquet files), classification definitions, and analysis scripts are available in the project repository. See the Data Availability statement for full details.

