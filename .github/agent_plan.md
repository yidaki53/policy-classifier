# Agent Plan: Swedish Parliament Policy Classifier

This document is the initial, high-level plan for an autonomous agent (and the developer workflow)
to build a reproducible, auditable policy-classifier for motions/proposals in the Swedish Riksdag.
The plan is intentionally implementation-focused and maps directly onto repository structure and
existing scripts (e.g. `scripts/ingest.py`, `scripts/sync.py`, `scripts/classify.py`).

## Goals
- Retrieve all motions and proposals from the Riksdag API and persist them to a local SQLite DB.
- Maintain an append-only (immutable) lineage history for every transformation.
- Classify each policy into ideological categories (`far right`, `right`, `centre right`, `centre`,
  `centre left`, `left`, `far left`) using transparent, logic-first rules with an interpretable
  AI component where needed. Allow multi-label assignments with strength weights.
- Compute party-level ideological profiles from weighted policy assignments.
- Produce publication-grade figures and a modular academic manuscript (each section in its own file).
- Enforce strict typing with Pydantic for all I/O and intermediate representations.
- Provide a reproducible CLI and tests so all analyses run without interactive agents.

## Core Design Principles
- Deterministic first: encode as much domain logic as explicit, reviewable rules.
- Transparency: all scores, weights and decisions must be auditable and reproducible.
- Modularity: clear separation between ingestion, cleaning, scoring, classification, aggregation, and viz.
- Reproducibility: seeds, versions, and data lineage tracked in the SQLite database.
- Interoperability: Pydantic models as canonical types for records and transforms.

## Data Ingestion & Storage
1. Riksdag retrieval agent
   - Use `scripts/sync.py` as the periodic sync entrypoint (or `scripts/ingest.py` for samples).
   - Retrieve motions, proposals, metadata (date, authors/party, text, motion type, url, motion id).
   - Only insert rows that are not already present (by unique motion id + last_updated); mark unsynced rows.
2. SQLite schema (high-level tables)
   - `motions`: raw retrieved JSON + parsed fields
   - `motions_text`: normalized text (title, body, cleaned tokens)
   - `parties`: canonical party metadata
   - `classifications`: one row per (motion_id, category, strength_score)
   - `lineage`: append-only table tracking operations `{op, timestamp, input_checksum, output_checksum, note}`
   - `sync_log`: records of API sync runs
3. Weighting logic
   - Add `retrieved_at` and `motion_date`; compute recency weight `w_recency = f(years_from_present)`.
   - Normalise weights so aggregates remain interpretable.

## Political Categories (Immutable Definitions)
- Store category definitions in a single YAML file under `definitions/political_spectrum.yaml`.
- This file is treated as immutable for a given release and versioned in git. For transparency,
  include human-readable rules and example motions for each category.

## Classification Approach
1. Rule-based deterministic scorer (first pass)
   - Create deterministic scoring rules (e.g., keywords, pattern matches, propositional logic)
   - Output: candidate categories with base scores and provenance (matching rule ids)
2. Interpretable AI model (second pass, optional)
   - Train a simple, explainable model (e.g., logistic regression, calibrated Naive Bayes, or
     a small gradient-boosted tree with SHAP explanations) on labeled examples.
   - Use model outputs as additional evidence; combine with rule scores to produce final strengths.
3. Multi-label & strength
   - Each motion can map to multiple categories; store `strength_score` in `[0,1]` per category.
   - Provide thresholding and soft-aggregation options.

## Aggregation to Party Profiles
- For each party, compute weighted sums over all motions they authored or sponsored.
- Use recency-weighted sums: S_party[c] = sum_over_motions(party,m) w_recency(m) * strength_score(m,c)
- Normalize by total weighted motions to produce a compositional profile (fractions per category).

## Lineage & Verification
- Every ETL/transformation writes a lineage row with input/output checksums and the code version.
- Add a `fixtures/` folder with small, deterministic test datasets and unit tests.
- Provide a `scripts/verify.py` that reproduces a deterministic run and asserts checksums.

## Manuscript & Visualization
- Follow the modular manuscript pattern used in other projects: each section in `manuscript/sections/`.
- Include `manuscript/Makefile` targets for `pdf`, `docx`, and `html`; keep figure generation code
  in `scripts/visualize.py` and `visualization/`.
- Candidate journals (method + political science fit): `Political Analysis`, `Party Politics`,
  `Legislative Studies Quarterly`, `European Journal of Political Research` (choose final venue early).

## Testing, CI, and Reproducibility
- Unit tests for ingestion, normalization, scoring, and aggregation (`tests/` exists already).
- A reproducible dockerfile or `requirements.txt`/poetry for environment pinning.
- CI: `pytest` + a lightweight verification job that runs `scripts/ingest.py --sample` and `scripts/classify.py`.

## Implementation Roadmap (first-pass tasks)
1. Finalize and version `definitions/political_spectrum.yaml` (immutable for release).
2. Implement robust Riksdag sync: `scripts/sync.py` -> write to `db/swedish_parliament.db`.
3. Add Pydantic models in `models/models.py` for all DB rows and pipeline records.
4. Implement deterministic scorer `classifier/scorer.py` with provenance logging.
5. Implement classification aggregation and party profile computation `analysis/aggregate.py`.
6. Add lineage tracking to `db/schema.py` and ETL steps.
7. Create manuscript skeleton under `manuscript/` with separate section files.

## Acceptance Criteria
- Able to run a sample sync and produce a reproducible party profile CSV.
- Every classification decision stores provenance and can be re-evaluated deterministically.
- Pydantic validation guards all external inputs/outputs.
- Manuscript skeleton exists with modular files and a build target to render PDF.

---

Notes:
- This file is a starting point; once accepted we will create concrete issues/PRs for each roadmap item.
- The `luck_vs_politics` repository demonstrates the manuscript modularity and build targets to emulate.
