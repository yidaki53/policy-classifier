
# From parliamentary claims to parliamentary conduct: multimodal estimation of Swedish party policy profiles from motions, speeches, and votes


# Abstract

Most public debate about party ideology relies on what parties say in manifestos or campaign messaging. At the same time, citizens and researchers need tools that track whether parliamentary conduct aligns with those claims. Existing computational pipelines often emphasize one modality (text or votes). They also optimize predictive performance without preserving transparent audit trails across the full workflow. We address this gap with a reproducible, parquet-first framework that estimates party policy positioning from three connected channels: motions, speeches, and roll-call votes.

Our objective is to operationalize a behavior-based ideology metric grounded in observed parliamentary activity. In plain terms, we estimate where parties stand by looking at what they formally propose, how they publicly argue, and how they actually vote. The pipeline starts from a transparent deterministic rule baseline (fixed and inspectable decision rules). It then adds embedding, zero-shot, and transformer signals in an ensemble design (a weighted combination of model families). Finally, it aligns modalities through explicit linkage and fairness-constrained optimization.

On the current full corpus, the workflow covers `n=202925` motions (1971-2024), `n=991235` speech-category rows corresponding to `n=141605` unique speeches (2014-2026), and `n=21464` unique roll-call vote events (1993-2026). With full speech-action linkage in the final stage, party-level consistency outputs are exported as auditable parquet artifacts. In labeled speech evaluation (`n=2656`), the baseline evaluation path yields accuracy `0.2033`; baseline NLL is `2.1535`, with calibration NLL `1.9221` (temperature) and `1.7115` (isotonic). The integrated hybrid ensemble used for exploratory comparison is materially stronger (`0.784` post-active-learning test accuracy on the aligned hybrid experiment), but the manuscript keeps the baseline path as the primary anchored claim to preserve conservative interpretation. Recency-weighted and lead-lag analyses provide party and parliament trajectories over time, and SARIMAX model selection is tracked through saved trial artifacts for reproducible forecasting diagnostics.

Linkage diagnostics (latest production refresh): full linkage coverage is achieved by design (`n=141605` linked rows), with confidence-stratified composition. In `output/analysis/speech_action_link_confidence_summary.json`, graph-signatory links are `n=95822` (`67.7%`), existing-reference links are `n=20841` (`14.7%`), heuristic fallback links are `n=12576` (`8.9%`), and structural high-confidence links are `n=12366` (`8.7%`). Action counts are near balanced (vote `n=71403`, motion `n=70202`).

Across the manuscript, the story runs from problem to evidence to implication. We define the measurement gap, implement an auditable multimodal estimator, test robustness under linkage and calibration uncertainty, and report descriptive party-level contrasts with explicit non-causal boundaries. The contribution is both methodological and empirical: a transparent parliamentary analysis stack that researchers can update, inspect, and stress-test as new data arrive. We use external references for triangulation (a directional comparison check), not as definitive ground truth, because statement-based benchmarks may diverge from realized parliamentary action.

The next section states the research question and comparative frame that govern these empirical claims.


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

We focus on Sweden because the parliamentary record offers unusually high institutional traceability for this research objective. We can observe party-level behavior consistently across motions, plenary speeches, and roll-call voting, all tied to a transparent legislative process. This setting reduces ambiguity about where parties make claims and where they record actions. That clarity is necessary because our core goal is to compare political speech with parliamentary conduct, not to maximize cross-country breadth.

We use official Riksdag open data as the primary source because each modality contributes a distinct inferential role that no single source can replace. Motions capture formal policy proposals. Speeches capture rhetorical framing and agenda emphasis. Votes capture enacted parliamentary choices under institutional constraints. The choice to combine these sources is not cosmetic. We need this combination to test whether observed ideology depends on what we measure as statement, proposal, or action.

The modeling strategy is deterministic-first and multimodal by design. We keep deterministic rules as the baseline because they provide inspectable decision logic and stable behavior across reruns. That stability is essential for auditability in an academic setting. We then add embedding, zero-shot, and transformer-based components as controlled extensions. These components improve coverage of linguistic variation and contextual nuance where fixed rules are too brittle. This layered design trades some simplicity for better representational capacity while retaining a transparent baseline for comparison.

To keep this design auditable, we treat major methodological choices as explicit tradeoffs rather than hidden defaults. We choose parquet-first processing over database-centric reporting workflows to preserve reproducible snapshot semantics and faster columnar analysis. We choose deterministic rules as the first layer to maximize inspectability, then add learned components to reduce linguistic brittleness. We choose fairness-constrained linkage over unconstrained matching to reduce party/time coverage artifacts, even when this increases engineering complexity. We choose descriptive inference boundaries over causal language because the current evidence structure is observational and linkage-sensitive.

The core architecture can be summarized as "rules for reliability, models for flexibility." In practical terms, we first use fixed rules that can be checked line by line, and then add learned models to catch meaning that appears in varied wording. We choose this hybrid setup instead of a single black-box model because the goal is to measure and explain party behavior, not only to maximize prediction scores. The tradeoff is straightforward: the system is more complex to build, but easier to audit and better at handling language variation.

For interpretive consistency, we define the core reported constructs once and apply them uniformly. Ideology index denotes the party-level location implied by category-weighted modality outputs (motions, speeches, votes). Consistency denotes cross-modality agreement after alignment and aggregation. We report it as a descriptive index, not a causal parameter. Fulfillment denotes the observed share of speech-linked policy pathways that proceed to vote-side action under current linkage assumptions. Contradiction denotes modeled mismatch intensity between speech-side positions and action-side outcomes. We use it as a diagnostic component of the broader consistency/fulfillment interpretation, not as a standalone causal estimate.

Because these terms can be confused, we keep their roles separate. Ideology index answers "where does this party appear to stand, based on observed behavior?" Consistency answers "how similar are the signals across speech, motions, and action records?" Fulfillment answers "of what is argued in speech, how much continues into linked action pathways?" Contradiction answers "how strongly do speech-side and action-side signals point in different directions?" None of these metrics is a moral grade or a causal estimate. Each is a structured summary of observed behavior under explicit assumptions.

The ensemble and alignment flow follows a fixed sequence. First, deterministic, embedding, zero-shot, and transformer components score motions. Next, a meta-learner combines those signals. Then explicit linkage joins speech outputs with vote-side records. Finally, the pipeline aggregates aligned outputs into party-year and party-topic-year diagnostics for consistency and fulfillment.

We distinguish two model-quality references in the workflow. The baseline evaluation path (`scripts/evaluate_ensemble.py`, default `models/ensemble_meta_clf.pkl`) is the conservative anchor used for manuscript claims and calibration reporting. The integrated hybrid stack (`scripts/train_hybrid_ensemble.py`, `models/hybrid_ensemble_meta_clf.pkl.zst`) adds zero-shot and transformer probability features on top of BERT CLS and base features, and is treated as an exploratory comparison rather than the primary manuscript anchor. That separation keeps the interpretive frame stable even when the hybrid stack achieves higher held-out accuracy.

Calibration and uncertainty checks are built in at each stage because model confidence is easy to over-interpret. In this study, a confidence score means how sure the model is internally. It does not guarantee factual correctness. We therefore test whether confidence is aligned with observed correctness on labeled data. We also run sensitivity checks to see whether headline conclusions remain similar when linkage and weighting settings are changed.

The latest stratified error pass (`scripts/run_stratified_sample.py`, UTC window `2026-06-03T13:15:06Z` to `2026-06-03T13:16:17Z`) reinforces this uncertainty policy. We observed three practical failure classes: rhetorical inversion (a speaker attacks an opposing position but inherits its lexical signal), low-margin centrist assignments in procedural/governance exchanges, and residual markup noise in historical text. These are not edge-case curiosities. They are expected consequences of lexical overlap and source-quality variation in parliamentary records. For that reason, we treat speech-level predictions as intermediate evidence and anchor interpretation on aggregated party-level diagnostics with explicit sensitivity checks.

We use fairness-constrained linkage and calibration diagnostics because speech and vote evidence are not uniformly available across parties, years, and topics. Without these controls, apparent ideological shifts can become artifacts of missingness or linkage imbalance rather than meaningful political change. We therefore report descriptive estimates with uncertainty diagnostics and avoid causal claims.

The fairness-constrained linkage step is especially important for interpretation. Linkage means connecting speeches to later parliamentary action records so speech and action can be compared directly. If some parties or time periods are easier to link than others, a simple comparison can look different just because the data are uneven. Rebalancing and confidence-level breakdowns are therefore used as bias controls. They do not remove all uncertainty, but they lower the risk that apparent party differences are mainly linkage artifacts.

Missing-pathway handling is also explicit. In the consistency workflow (`scripts/analyze_consistency_trends.py`), the fulfillment component is configurable through `--fulfillment-fill` and is used when a party-topic observation has no available motion pathway after linkage. This is a fairness tradeoff rather than a hidden default: imputing a neutral or low value avoids mechanically penalizing parties with sparse linkage support, but it also introduces an assumption that must be tested. We therefore treat fulfillment-imputation sensitivity as part of model uncertainty and report results as descriptive estimates conditional on the selected setting. In the latest refresh, no party-level missing-motion pathways remained after linkage aggregation, so fill sensitivity (`0.0` vs `0.5`) produced identical party-level consistency outputs; this is reported as a verification result rather than assumed ex ante.

We selected the reference set for function rather than citation breadth. Foundational text-as-data and political scaling studies define the measurement assumptions used here. Parliamentary NLP work motivates modality-specific choices in debate text processing and interpretation. We use external benchmarks such as CHES for triangulation only, because expert-survey ideology captures perception and positioning at a different abstraction level than behavior observed in parliamentary records.

Benchmark choice also carries an interpretation tradeoff. External references such as the Chapel Hill Expert Survey (CHES) improve directional plausibility checks, but they do not provide a direct ground-truth target for parliamentary behavior trajectories in this pipeline. Agreement with CHES can strengthen confidence that party ordering is not arbitrary; disagreement can be substantively informative when speech and vote behavior diverge from expert-position estimates under coalition or agenda constraints. We therefore treat benchmark alignment as a validity signal, not a correctness oracle.

The executable implementation maps directly to this rationale. Deterministic-first scoring and hybrid aggregation are defined in `scripts/classify.py` and `src/swedish_parliament_policy_classifier/classifier/scorer.py`. Linkage and fairness controls are implemented in linkage and consistency-tuning scripts under `scripts/`. Recency and temporal diagnostics are generated by `scripts/analyze_recency_weighted_trends.py` and SARIMAX artifacts exported under `output/analysis/`. This script-to-artifact mapping is the operational mechanism that keeps methods claims auditable.

Finally, we treat all outputs as conditional on this design envelope. The method supports reproducible descriptive inference about parliamentary behavior, not universal claims about "true" ideology. Result quality depends on dictionary design, category definitions, linkage quality, and classifier calibration. The artifact chain explicitly documents and tests each of these components.

With these design choices justified, the next section reports the empirical patterns and concrete examples that follow from this measurement framework.


# Results

This section reports what the pipeline estimates in practice and where observed party differences are strongest. The focus is empirical rather than procedural. We present modality-sensitive contrasts, fulfillment patterns, and consistency contrasts using the current parquet artifacts.

How to read this section: each reported metric summarizes observed parliamentary behavior under clearly stated assumptions. A higher consistency value means stronger agreement between what a party proposes, says, and does in linked action records. A higher fulfillment value means a larger share of speech-linked issue pathways that continue into vote-side action records. A higher contradiction value means speech-side and action-side signals point in more different directions. Use these metrics as comparison tools, not as causal effect estimates.

## Key Visual Evidence
The figures below show headline outputs directly used for main-text interpretation.

![Consistency vs Fulfillment (updated 2026-05-31T02:49:40Z)](../output/manuscript/figures/figure_consistency_vs_fulfillment.png){ width=90% }

![Parliament Direction Over Time (updated 2026-05-31T02:49:40Z)](../output/manuscript/figures/figure_parliament_direction_over_time.png){ width=90% }

We moved intermediate, process-oriented figures to the appendix to keep the Results section focused on headline evidence.

## Corpus Coverage and Model Quality

On the current full corpus, the workflow covers `n=202925` motions (1971-2024), `n=991235` speech-category rows corresponding to `n=141605` unique speeches (2014-2026), and `n=21464` unique roll-call vote events (1993-2026). With full speech-action linkage in the final stage, party-level consistency outputs are exported as auditable parquet artifacts. In labeled speech evaluation (`n=2656`), baseline accuracy is `0.2033`; baseline NLL is `2.1535`, with calibration NLL `1.9221` (temperature) and `1.7115` (isotonic). Recency-weighted and lead-lag analyses provide party and parliament trajectories over time, and SARIMAX model selection is tracked through saved trial artifacts for reproducible forecasting diagnostics.

These figures indicate informative but uncertain signal. The baseline evaluation path remains intentionally conservative for manuscript claims, while the integrated hybrid ensemble is materially stronger on held-out test data and is retained as an exploratory comparison rather than the anchor for the manuscript's core quantitative claims. We therefore interpret all downstream contrasts as descriptive evidence rather than definitive recovery of a single hidden ideology value.

Across hypotheses, the results are consistent with modality-sensitive ideology measurement under a descriptive interpretation. Party-level profiles differ across motions, speeches, and vote-linked action channels. Speech-action consistency also varies across parties after linkage constraints. Fulfillment diagnostics add information beyond aggregate consistency alone.

## Cross-Modality Contrasts

(See Figure 1, Consistency vs Fulfillment, generated by `scripts/analyze_consistency_trends.py`; see Figure 3, Party Modality Overlay, generated by `scripts/generate_manuscript_overlay.py`; see Figure 8, Three-way Divergence, generated by `scripts/speeches_analysis.py`.)

Substantively, this means no single channel can be treated as a complete proxy for party ideology. Motion-side evidence can reflect formal agenda setting and coalition strategy. Speech-side evidence can reflect rhetorical framing and constituency signaling. Vote-linked action can reflect final institutional bargaining constraints. When these channels converge, confidence in the descriptive estimate increases. When they diverge, the divergence itself becomes a meaningful analytical result rather than a nuisance to suppress.

One concrete speech-level example shows how the classifier surfaces interpretable ideological signal from parliamentary language. Speech `c6c44eb9-b09c-e411-9412-00262d0d7125` (party `V`) is assigned category `left` with normalized weight `1.00` and confidence `1.00` in the speech classification artifact (`speech_classifications_with_rhetoric_full.parquet`). The speech text includes the statement: "Vi har redan varit med om Skånepolisens registrering av romer, och nu är det ett kvinnoregister ... Den här utvecklingen måste stoppas." This is not presented as proof of latent ideology on its own, but as an auditable instance of category assignment that can be traced back to source text and model output.

This example is included for transparency, not anecdotal persuasion. It demonstrates how a category assignment can be audited from source text to model output. This matters for reproducibility and interpretive discipline. A single speech cannot establish party-level ideology, but it can show whether the pipeline produces traceable and linguistically plausible intermediate outputs before aggregation.

(See Figure 8, Three-way Divergence, generated by `scripts/speeches_analysis.py`.)

Promise-fulfillment contrasts are substantively visible in the current summary table. In `output/analysis/promise_fulfillment_party_summary.parquet`, `SD` has `pct_speech_motion_vote = 0.3526` while `V` has `0.1787`; `V` shows `pct_speech_motion_no_vote = 0.0921`. These differences illustrate why fulfillment diagnostics are retained as a separate axis instead of being collapsed into one aggregate consistency score.

We interpret the fulfillment contrast as a pathway diagnostic. It asks whether issue emphasis in speech is followed by linked formal action at different rates across parties and issue domains. This pattern does not imply direct legislative causation from speech to votes. Instead, it quantifies how often speech-side attention appears in pathways that continue toward action records.

Consistency contrasts remain modest in absolute spread but informative for ranking and comparison. In `output/analysis/consistency_score_party.parquet`, `M` records `consistency_score = 0.5454` and `motion_pathway_fulfillment = 0.8882`, while `L` records `consistency_score = 0.5112` and `motion_pathway_fulfillment = 0.5804`. The ranking difference is interpreted as descriptive signal under linkage and calibration assumptions, not as evidence of causal party effects.

The consistency contrast complements fulfillment by focusing on agreement structure rather than endpoint rates. Two parties may display similar aggregate consistency while differing sharply in where that consistency comes from. For example, one party may show stable vote alignment but variable speech framing. For this reason, we interpret consistency and fulfillment jointly. Consistency indicates coherence across channels, while fulfillment indicates pathway continuation from speech-linked records into action-linked records.

(See Figure 1, Consistency vs Fulfillment, generated by `scripts/analyze_consistency_trends.py`.)

We keep classifier quality and substantive interpretation separate throughout. We treat calibration choices, linkage fairness constraints, and uncertainty intervals as sensitivity controls that bound interpretation. The baseline evaluation path is the one cited in the main manuscript claims; the integrated hybrid ensemble is discussed as a higher-performing exploratory variant, not as the primary claim anchor. All cross-party contrasts remain descriptive rather than causal. We use external benchmarks for directional triangulation only, because statement-based benchmarks can diverge from observed parliamentary action.

This separation between model quality and substantive claim strength is central to the manuscript's inferential stance. Better classifier metrics increase confidence that labels are coherent under the chosen category system, but they do not automatically justify stronger causal claims about party intent or policy consequences. Conversely, modest classifier performance does not invalidate all comparative diagnostics if uncertainty is explicitly modeled and interpretation remains bounded.

Recency-weighted party and parliament summaries continue to support the same interpretation boundary and are exported to `output/analysis/recency_weighted_party_scores.parquet`, `output/analysis/recency_weighted_parliament_timeseries.parquet`, and `output/analysis/recency_weighted_summary.json`.

We use recency weighting to answer a specific temporal question. Do contemporary party positions reflect information that is closer in time to current parliamentary behavior, rather than an equal average of distant historical periods? This improves interpretability for present-facing comparisons. It also introduces an explicit tradeoff. Short-term volatility can carry more influence than long-run structural stability.

(See Figure 2, Parliament Direction Over Time, generated by `scripts/analyze_consistency_trends.py` and `scripts/analyze_recency_weighted_trends.py`.)

## Robustness and Interpretation Limits

The speech-to-motion linkage uses rel_dok_id-to-betankande bridging with graph-direct and fallback strategies. Full linkage coverage is achieved by design: `n=141605` speeches are assigned a motion or vote-side candidate (`coverage=1.0000`). Evidence quality is therefore interpreted through confidence composition rather than raw coverage alone. In `output/analysis/speech_action_link_confidence_summary.json`, `n=95822` links (`67.7%`) are graph-signatory, `n=20841` (`14.7%`) are existing-reference links, `n=12576` (`8.9%`) are heuristic fallback links, and `n=12366` (`8.7%`) are structural-high links. Benchmarks are used for directional triangulation only, and election-runup summaries are treated as descriptive trend diagnostics rather than outcome forecasts.

The linkage diagnostics clarify how much of the speech corpus enters cross-modality comparison at different confidence levels. Higher-coverage linkage increases representativeness, while confidence-level splits provide a direct view of robustness under stricter versus looser matching criteria. Readers can then evaluate whether key comparisons are stable only in permissive linkage settings or remain visible under stricter thresholds.

Stage-3 stratified error review also identified three recurring failure modes that bound interpretation of speech-level labels. First, rhetorical inversion cases appear when opposition speeches quote or attack right-coded issue language; this can induce rightward predictions even for left/center-left speakers (for example, speech IDs `277ee5c2-d93f-f111-bf21-6805cafeabf9` and `82a2f18c-2482-e511-942d-00262d0d0c40` in `stratified_classification_report.md`). Second, governance/procedural debate often yields low-margin centrist assignments with near-tied alternatives. Third, occasional markup contamination (for example `STYLEREF ... MERGEFORMAT`) remains in source text previews and can perturb token evidence. We therefore treat individual speech labels as auditable but noisy intermediates and prioritize party-level aggregates and sensitivity checks for substantive claims.

Fulfillment-imputation sensitivity was also checked directly in the current artifact set. Recomputing consistency outputs with `--fulfillment-fill` set to `0.0` and `0.5` produced identical party-level consistency estimates in this refresh (`rho = 1.000`, max absolute delta `0.000`), because no party-level missing-motion pathways remained after aggregation.

Current metric anchors from this workflow include `n=403` rows in party-topic-year fulfillment and expected-contradiction aggregates, and `n=480` successful SARIMAX trials for monthly model selection.

Taken together, these results support a bounded empirical claim. Ideology-related party contrasts are detectable in multimodal parliamentary evidence, but their magnitude and ranking depend on linkage assumptions, model calibration, and category design. The appropriate reading is comparative and diagnostic, not definitive or causal.

(Methods pointer: `scripts/classify.py` and `src/swedish_parliament_policy_classifier/classifier/scorer.py` define the deterministic-first scoring baseline used in this workflow.)

A final synthesis helps bound interpretation. Robust findings are those that persist across modality comparisons and linkage-sensitivity diagnostics, including modality-sensitive party contrasts and cross-party variation in consistency/fulfillment. Suggestive findings include rank-order differences with modest absolute spread and outcomes that depend more strongly on calibration or weighting choices. Provisional findings include trend magnitudes in settings with higher fallback linkage reliance or model-family uncertainty. This hierarchy is used to keep empirical claims proportional to current evidence quality.

For first-time readers, see the compact plain-language guide in the Appendix section "How to read the metrics."

Run provenance for the latest full-chain recency and robustness execution remains anchored to `scripts/extract_motion_signatories.py`, `scripts/tune_link_rebalance_fair_ga.py`, `scripts/link_all_speeches_to_action.py`, `scripts/compute_ideology_axis_alignment.py`, `scripts/score_say_vs_do_contradiction.py`, `scripts/tune_consistency_wrangling_fair_ga.py`, and `scripts/analyze_consistency_trends.py`. The UTC timestamp is `2026-05-31T20:47:30Z` and outputs are written under `output/analysis/`.

These findings establish the empirical basis for the manuscript and motivate the final conclusion on what this framework can and cannot claim.


# Conclusion

This section closes the manuscript's central story. We began with a societal and academic gap: difficulty evaluating whether parliamentary conduct matches party claims at scale. We then built and tested a reproducible multimodal estimator that remains inspectable end to end. The significance is that this workflow turns fragmented parliamentary signals into a single auditable measurement process with explicit uncertainty boundaries.

In practical terms, the manuscript contributes more than party-level scores. It also makes those scores inspectable from origin to interpretation. Readers can see where information enters the pipeline, how signals are transformed, where uncertainty is introduced, and which assumptions materially affect comparative conclusions. This transparency is essential for policy-relevant computational work, where it often matters as much as raw predictive performance.

The deterministic chain matters because it keeps the analysis traceable from raw parquet inputs through linkage, classification, annual summaries, recency weighting, and SARIMAX time-series modeling. This makes party comparisons and election-runup checks reproducible rather than purely model-dependent. It also enables sensitivity checks for linkage coverage, window definitions, and seasonal specification. In this framing, the core contribution is a behavior-based operational metric of party ideology under stated assumptions, not recovery of a single externally defined ground-truth scale.

This distinction should be explicit. A behavior-based metric asks, "what pattern is visible in observed parliamentary records under declared assumptions?" It does not ask, "what is the true ideology of a party in a metaphysical sense?" By preserving that distinction, the manuscript avoids overclaiming while still offering a useful comparative instrument for political analysis.

The results provide descriptive evidence consistent with all three hypotheses under the manuscript's non-causal frame: modality-sensitive profiles, party-varying say-do consistency, and added explanatory value from fulfillment and contradiction diagnostics. Where model quality is discussed, the baseline evaluation path remains the manuscript anchor, while the integrated hybrid ensemble is stronger on held-out test data and is treated as an exploratory comparison rather than the core claim source.

The empirical value is cumulative rather than singular. We do not interpret any single figure or score as definitive. Instead, confidence comes from convergence across modalities, stability under sensitivity checks, and traceability in the artifact chain. Where those conditions are weaker, the manuscript states reduced confidence and narrows interpretation accordingly.

These estimates describe political text-action alignment and are not interpreted causally. Baseline speech-classifier accuracy (`0.2033`) remains below ceiling, so readers should treat modality-level contrasts as exploratory measurement outputs rather than definitive classifications. The stronger integrated hybrid ensemble is reported separately and does not replace the baseline anchor used for manuscript claims. Label-space metrics benchmarked to Britannica-based categories do not, by themselves, validate an external latent-ideology criterion. We therefore interpret external benchmark disagreement cautiously, because statement-based references (including manifesto-oriented sources) can diverge systematically from observed parliamentary conduct.

Uncertainty is also inherited from upstream linkage and classification. Current artifacts (`output/analysis/speech_action_link_confidence_summary.json`) report full linkage coverage and confidence stratification. Although the latest run is no longer mainly fallback-driven, confidence strata still vary by party. Dictionary design, model-family weighting, and linkage rebalance choices can shift magnitudes, so substantive interpretation should remain tied to sensitivity checks. Forecast components function as model-fit diagnostics for trend characterization, not as validated policy-outcome prediction tools.

These caveats are not a weakness of the study design. They are a methodological safeguard. Parliamentary language and action records are complex social data, and explicit uncertainty treatment is necessary to keep claims scientifically proportional to evidence quality. The manuscript therefore treats uncertainty reporting as part of the contribution, not merely as a limitations paragraph.

If substantive conclusions remain stable across linkage-confidence strata, model-family variants, and uncertainty intervals, confidence in the latent-ideology operationalization increases. In the current refresh, structural-vs-all stability still shows measurable drift (abs max delta 0.048 in `output/analysis/link_strata_stability_summary.json`). We therefore interpret outputs as comparative diagnostics under explicit modeling assumptions, not as a fully validated single latent-trait estimate.

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

![Consistency-Fulfillment vs External Benchmark (Party-Year) (updated 2026-06-03T21:53:40Z)](../output/manuscript/figures/figure_consistency_fulfillment_vs_benchmark_party_year.png){ width=100% }

![Party Modality Overlay (updated 2026-05-31T21:41:29Z)](../output/manuscript/figures/figure_modality_overlay_by_party.png){ width=100% }

![Motion Category Distribution (updated 2026-06-03T21:55:33Z)](../figures/manuscript/pie_chart_categories.png){ width=90% }

![Party Motions Stacked (updated 2026-06-03T21:55:33Z)](../figures/manuscript/party_motions_stacked.png){ width=90% }

![Voting Cohesion Time Series (updated 2026-06-03T21:56:13Z)](../figures/voting/party_cohesion_timeseries.png){ width=100% }

![Three-way Divergence (updated 2026-06-03T13:16:17Z)](../figures/three_way/divergence_speech_vs_combined_significance.png){ width=100% }

![Speech Profiles Heatmap (updated 2026-06-03T21:51:00Z)](../figures/speeches/speech_profiles_heatmap.png){ width=100% }


# Data Availability

All data and metadata underlying the findings reported in this manuscript are available within the project repository and its reproducible artifact directories. Source parliamentary records are retrieved from official Swedish Parliament open-data endpoints and normalized into compressed parquet datasets under `data/parquet/`, `data/speeches/parquet/`, `data/betankande/parquet/`, and `data/votering/parquet/`. Derived analysis tables used for results and figures are available under `output/analysis/`, and generated figure assets are available under `output/manuscript/figures/` and `figures/`.

All scripts required to reproduce ingest, classification, linkage, analysis, and figure generation are included under `scripts/` and are executed in a pinned Python environment using `uv`. The exact rendering/build context for the manuscript is exported at build time to `manuscript/build/manuscript_context.json`, and journal-readiness checks are exported to `manuscript/build/journal_requirements_report.json`.

No participant-level restricted data are introduced by this project; all primary inputs originate from publicly available parliamentary materials.

The full reproducible project is publicly accessible at `https://github.com/yidaki53/policy-classifier`. Submission and production versions should cite the exact release tag and commit hash used for manuscript generation.

For production handoff, the recommended archival path is to create a versioned release snapshot and archive it in a long-term repository service (for example, a Zenodo-linked GitHub release). This preserves the exact manuscript-state code and artifacts and provides a persistent accession identifier for citation without changing the underlying access pathway described above. The archived record should include the release tag, commit hash, artifact directory inventory, and manuscript build timestamp used in the submitted version.

Numeric reporting policy: JSON artifacts in `output/analysis/` preserve machine-precision float values (IEEE754). In manuscript prose and figure captions, percentages are rounded for readability (typically to one decimal place unless otherwise stated). Where rounded text differs from full-precision values, the full-precision artifact is the reproducible reference.


# Acknowledgments

The authors thank the Swedish Parliament (Riksdagen) for providing open-access data through the Riksdag Open Data API, which underpins all corpus materials used in this analysis.

## Author Contributions (CRediT)

Robin Oberg: Conceptualization, Methodology, Software, Data Curation, Formal Analysis, Visualization, Validation, Writing - Original Draft, Writing - Review and Editing.

This manuscript reports a single-author study. Contributor roles are declared using the CRediT taxonomy for submission metadata alignment.

No external funding was received for this study. The authors received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

The authors declare no competing interests.

All source data used in this manuscript are publicly available via the Riksdag Open Data API (data.riksdagen.se). Processed analysis artifacts (parquet files), classification definitions, and analysis scripts are available in the project repository. See the Data Availability statement for full details.

