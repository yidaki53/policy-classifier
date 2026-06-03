---
section_id: "02_question"
section_title: "Question"
objective: "Define the central research question, units, and comparative frame for the analysis chain."
required_inputs:
	- "Current modality coverage and outputs from analysis pipeline."
	- "Current definition of comparative targets (parties, topics, years, modalities)."
required_outputs:
	- "Primary research question text aligned to implemented methods and outputs."
required_metrics:
	- "none"
required_figures_tables:
	- "none"
provenance_requirements:
	- "Question wording must be compatible with reported result artifacts and metrics."
update_triggers:
	- "Change in analytical objectives or target comparisons."
	- "Addition/removal of major modalities or tasks."
owner: "manuscript-agent"
status: "active"
last_updated_utc: "2026-05-29T08:05:00Z"
---

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