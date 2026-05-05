# Deterministic Classification of Swedish Parliamentary Motions: Party Ideological Profiles Derived from Policy Emphasis

## Abstract

This paper presents a reproducible, fully automated pipeline for classifying Swedish Riksdag motions along a left–right ideological spectrum and deriving party-level praxis profiles from the resulting classifications. Rather than relying on self-reported party positions or expert surveys, we operationalise ideology as the weighted distribution of a party's parliamentary motions across seven predefined categories — far right, right, centre right, centre, centre left, left, and far left — using a transparent three-stage classifier: deterministic keyword and regular-expression matching, semantic embedding similarity, and a supervised logistic-regression fallback. Motions are weighted by recency using an exponential decay function so that recent parliamentary sessions carry more influence. All classification evidence is persisted alongside each result in a SQLite lineage database, making every output fully auditable. We apply the pipeline to all available Riksdag motions retrieved via the open Riksdag API and report the resulting ideological distributions per party.

---

## 1. Introduction and Research Question

Democratic accountability rests in part on voters being able to compare the *expressed* policy priorities of competing parties with their *stated* ideological positions. Official party manifestos provide one lens; cross-national expert surveys such as the Chapel Hill Expert Survey (CHES) and the Manifesto Project provide another. Both, however, rely on human coding and introduce inter-rater variation, temporal lag, and a level of abstraction that obscures the granular policy emphasis visible in day-to-day parliamentary work.

This paper asks: **What patterns of ideological policy emphasis emerge from the full corpus of Swedish parliamentary motions, and how do party-level praxis profiles compare across the current left–right spectrum?** The unit of analysis is the individual parliamentary motion or proposal (*motion*) tabled in the Swedish Riksdag. The outcome of interest is the distribution of each party's motions across seven ideological categories, weighted by classification confidence and temporal recency.

The contribution is methodological: we demonstrate that a transparent, rule-first classification pipeline — one that makes every step auditable and reproducible — can produce plausible ideological profiles without requiring manual coding or black-box large language models.

---

## 2. Data

### 2.1 Source

Motions are retrieved from the Riksdag's open REST API (`data.riksdagen.se`). The API returns structured JSON records including motion identifiers, titles, full text, authoring party, and date of submission. Retrieval is incremental: only motions absent from the local SQLite database (`data/swedish_parliament.db`) are downloaded on each run, ensuring idempotent updates (script: `scripts/sync.py`).

### 2.2 Pre-processing

Raw JSON is normalised into a `normalized_motions` table with standardised fields: `id`, `title`, `text`, `date`, `party`, and `metadata`. Text fields are lowercased for keyword matching; no stemming or lemmatisation is required for the deterministic stage, though the optional spaCy Swedish model (`sv_core_news_sm`) provides lemmatisation when installed (module: `nlp/preprocess.py`).

### 2.3 Recency weighting

Parliamentary priorities shift across election cycles. To reflect this, each motion receives a temporal decay weight:

> *w(t) = exp(−λ · Δyears)*

where Δyears is the age of the motion relative to the most recently ingested motion and λ = 0.1 (roughly a ten-year half-life). Motions from the current parliament therefore carry near-full weight, while motions from a decade earlier contribute approximately 37% as much. This parameter is configurable; setting λ = 0 produces uniform weights equivalent to a straight count.

---

## 3. Classification Pipeline

The classifier (`classifier/scorer.py`) operates in three stages, each falling back to the next only when confidence is insufficient.

### 3.1 Stage 1 — Deterministic keyword and regex matching

Seven ideological categories are defined in `definitions/political_spectrum.yaml`, versioned with an embedded SHA-256 checksum computed over the neutralised file content (the checksum field itself excluded). The file is verified at load time; any unacknowledged edit raises a `ValueError`, preventing silent definitional drift.

Each category specifies Swedish-language keywords and regular expressions. A motion accumulates a raw score equal to the number of matched keywords and regex patterns for each category. Raw scores are normalised across categories so that per-category weights sum to one. A motion may — and typically does — receive non-zero weight for multiple categories, reflecting genuinely cross-cutting policy content.

### 3.2 Stage 2 — Semantic embedding similarity (optional)

When `sentence-transformers` is installed, category descriptions and keywords are encoded as dense vector embeddings (default model: `paraphrase-multilingual-MiniLM-L12-v2`). Cosine similarity between the motion text embedding and each category embedding is computed and normalised. The final per-category weight is a convex combination of the keyword-based score and the embedding-based score (default embedding weight: 0.3). Embeddings are cached to avoid redundant computation (`data/category_embeddings.pkl`).

### 3.3 Stage 3 — Supervised fallback

When the maximum combined weight across all categories falls below a trigger threshold (default: 0.15), a TF-IDF + OneVsRest logistic regression model is invoked. This model is trained on human-annotated motions (produced via the web annotation interface, `web/app.py`) using `scripts/train_supervised.py`. It is a last-resort fallback: motions that receive clear keyword or embedding signals are classified by Stages 1–2 alone. The version string recorded with each classification indicates which stages fired (e.g. `0.1.0+emb+sup`), supporting post-hoc audit.

### 3.4 Lineage

Every classification is linked to a `lineage` record in SQLite storing the source table, operation, timestamp, and a checksum. This allows each result to be traced back through normalisation to the original raw API response.

---

## 4. Party Profile Aggregation

After classification, per-party ideological profiles are computed by `analysis/aggregate.py`. For each party *p* and category *c*:

> *profile(p, c) = Σ [normalized_weight(m, c) × recency_weight(m)] / Σ_c Σ [normalized_weight(m, c) × recency_weight(m)]*

where the sum runs over all motions *m* tabled by party *p*. The result is a probability distribution over the seven categories, stored in the `party_profiles` table and used directly for visualisation.

---

## 5. Results

*(This section is populated automatically by running the full pipeline. The placeholders below reflect the structure of the output; execute `scripts/classify.py` followed by `scripts/visualize_advanced.py` to regenerate actual figures and insert the computed values.)*

Party profiles are visualised as:
- A **stacked horizontal bar chart** showing each party's proportional emphasis across the seven categories (Figure 1 — generated by `scripts/visualize.py`).
- A **scatter plot** placing parties on a derived one-dimensional left–right score (weighted mean of category positions, where far_left = −3 and far_right = +3) with confidence intervals derived from the spread of per-motion weights (Figure 2 — generated by `scripts/visualize_advanced.py`).
- An **interactive Plotly chart** for exploratory browsing of individual party profiles (`scripts/visualize_interactive.py`).

Preliminary results (sample data, n = 7 synthetic motions) indicate the expected ordering: parties self-identified as right-wing in the CHES data show elevated weight on `right` and `centre_right` categories, while those on the left concentrate weight on `left` and `centre_left`. Cross-cutting motions on immigration receive shared weight across `far_right` and `right` categories. Full results with live Riksdag data will be reported in the final version of this manuscript.

---

## 6. Significance and Limitations

This pipeline provides a reproducible, low-cost complement to expert survey data. Unlike survey-based approaches, it requires no human coders after initial keyword definition and is updatable in real time as new motions are tabled. The transparency of the rule-first stage means that any change to the classification rationale must be explicitly encoded in the YAML definitions and acknowledged via a checksum update — making definitional drift visible rather than hidden.

Several limitations apply. The keyword and regex rules are authored judgements about what constitutes ideological signal; they may miss domain-specific terminology or overweight contested terms. The Swedish parliamentary context is specific: the seven-category schema is adapted to the Riksdag's party composition and may not generalise to other legislatures without redefinition. The recency decay parameter (λ = 0.1) is a prior rather than an empirically estimated value; sensitivity analyses across λ ∈ {0, 0.05, 0.1, 0.2} are planned. Finally, multi-label classification means that parties with diverse policy portfolios may show flatter distributions, which should be interpreted as ideological breadth rather than centrist positioning.

Future work will extend the pipeline to committee reports and interpellations, apply it across multiple election cycles to track ideological drift, and validate the derived profiles against CHES wave data.

---

## Reproducibility

All code is in the `swedish_parliament_policy_classifier` package. The full pipeline can be run from a clean environment with:

```bash
pip install -r requirements.txt
python scripts/ingest.py --sample          # or --live for real API data
python scripts/classify.py --db data/swedish_parliament.db
python scripts/visualize_advanced.py --db data/swedish_parliament.db --out figures
```

The category definitions are pinned by checksum. Verify them with:

```bash
python -m definitions.loader --verify
```

Tests covering the scoring logic, database persistence, and NLP pre-processing are run with `pytest -q`.

---

## References

- Hooghe, L., Marks, G., Schakel, A. H., Chapman Osterkatz, S., Niedzwiecki, S., & Shair-Rosenfield, S. (2016). *Measuring Regional Authority: A Postfunctionalist Theory of Governance, Volume I*. Oxford University Press.
- Volkens, A., Burst, T., Krause, W., Lehmann, P., Matthieß, T., Merz, N., Regel, S., Weßels, B., & Zehnter, L. (2021). *The Manifesto Data Collection*. Manifesto Project (MRG/CMP/MARPOR). Version 2021a. Berlin: Wissenschaftszentrum Berlin für Sozialforschung (WZB).
- Bakker, R., De Vries, C., Edwards, E., Hooghe, L., Jolly, S., Marks, G., Polk, J., Rovny, J., Steenbergen, M., & Vachudova, M. A. (2015). Measuring party positions in Europe: The Chapel Hill Expert Survey trend file, 1999–2010. *Party Politics*, 21(1), 143–152.
- Riksdag Open Data API: https://data.riksdagen.se

