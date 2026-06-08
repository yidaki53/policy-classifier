---
name: swedish-parliament-policy-classifier-agent
scope: repo
repo: swedish_parliament_policy_classifier
stack:
  - python
  - parquet
  - transformers
  - ollama
  - graphify
---

# swedish_parliament_policy_classifier

- This file complements `.github/copilot-instructions.md`: keep repo-wide defaults there, keep agent-facing workflow policy here, and keep file-scoped rules under `.github/instructions/`.
- Focus on classifier pipeline, YAML definitions, Parquet-based data layer, and NLP preprocessing.
- Parquet-only policy: do not use SQLite/DB-backed workflows for analysis, training, or evaluation; read/write compressed Parquet (`.parquet` with zstd) under `data/parquet/` and `output/`.
- Academic-rigor policy: as work progresses, document methodology, assumptions, model/config choices, intermediate results, and figure provenance (script + input/output paths + timestamp) in repo docs/manuscript notes.
- Thermal-safe policy: default long-running scripts to conservative resource settings (CPU fraction ~0.25 or lower plus pacing sleeps), and require explicit opt-in for hotter configurations.
- Keep changes small and reversible. Prefer deterministic rules before ML.
- Preserve naming contracts and deterministic-first principle.
- Add or update tests in the same change as code.
- Validate with `uv run pytest -q` before submitting.
  - CI runs `uv pip install -e .`, a smoke import test, and `uv run pytest -q`.
- **For architecture questions, use the graphify skill (`/graphify`) FIRST.** Query `graphify-out/graph.json` via `graphify query`, `graphify explain`, or `graphify path` before reading source files directly. This saves tokens and provides cross-document context that file-by-file reading misses.
- Use relative paths only. Never use absolute paths.
- Update the graph after substantial edits with `graphify update .`.
- Use Graphify in AST-only mode for repo graph updates; do not use semantic extraction in this repository workflow.

## Fast Path: Save Tokens And Time

Use this order for almost every task:

1. Ask Graphify first.
2. Read only the smallest set of source files needed to edit.
3. Keep generated docs/data machine-readable with valid frontmatter.
4. Run the narrowest validation command first, then widen only if needed.

### Graphify-first workflow (required for architecture/navigation questions)

- Start with one of:
  - `graphify query "<question>"` for broad understanding.
  - `graphify explain "<file-or-concept>"` for one anchor.
  - `graphify path "<A>" "<B>"` for dependency chains.
- Only fall back to broad `rg`/multi-file reads if graph output is insufficient for an exact edit.
- Prefer scoped graph queries over opening `graphify-out/GRAPH_REPORT.md`.

### Frontmatter rules for generated files (required)

- Any generated Markdown file must begin with valid YAML frontmatter.
- Any generated YAML/Markdown metadata blocks must be valid YAML 1.2: use spaces (no tabs), consistent indentation, and simple scalar values.
- Keep frontmatter compact: include only keys needed for routing/automation.
- Prefer a stable `_agent_frontmatter` object with at least: `id`, `purpose`, `steward`, `edit_policy`, and `generator` for generated artifacts.
- If a generator updates a file that already has frontmatter, preserve existing fields unless there is an explicit migration.

### Minimal templates

Markdown:

```yaml
---
_agent_frontmatter:
  id: "<doc.id>"
  purpose: "<what this file is for>"
  steward: "<team-or-module>"
  edit_policy: "manual|generated_do_not_edit"
  generator: "<script path if generated>"
---
```

YAML:

```yaml
_agent_frontmatter:
  id: "<config.id>"
  purpose: "<what this file is for>"
  steward: "<team-or-module>"
  edit_policy: "manual|generated_do_not_edit"
  generator: "<script path if generated>"
```

## Using graphify to save tokens (MANDATORY for architecture questions)

Before reading multiple source files to understand the codebase, **always** try graphify queries first:

- `/graphify query "How do classification results flow from scorer to aggregate to visualization?"` — BFS traversal for broad context.
- `/graphify path "score_motion" "plot_party_ideology_heatmap"` — trace the exact dependency chain between two symbols.
- `/graphify explain "classify_speeches_parquet.py"` — inspect all connections to a specific module/function.

Only read source files directly when the graph answer is insufficient or you need to verify exact line numbers.

See `docs/GRAPHIFY_TOKEN_TIPS.md` for model-selection and cache configuration.

## Recent agent-run learnings (May 2026)

- Added targeted, high-confidence `if False:` import hints and `src/` re-exports to help static/semantic extractors resolve cross-module links.
- Introduced a single-file canonical exports module (`src/swedish_parliament_policy_classifier/exports.py`) to provide one literal import path for commonly-used symbols.
- Performed an aggressive cleanup (removed duplicate packaged scorer) and then restored a packaged scorer wrapper; experiments showed semantic extraction behavior varies by model.
- Observed Graphify inferred-edge regressions during iterative edits (baseline inferred=72 -> current ~96). This suggests extraction heuristics (AST vs semantic LLM) and model choice materially affect inferred links.
- Migrated primary data storage from SQLite to Parquet (`data/parquet/`). All analysis scripts should read from and write to Parquet.
- Treat SQLite (`data/swedish_parliament.db`) as legacy/migration-only; do not introduce new DB dependencies.

Recommendations for future agents:

- Prefer small, incremental changes and re-run `graphify update .` between edits to measure impact.
- Use `if False:` import hints near call-sites for high-confidence linking when refactors are costly.
- Consolidate canonical export paths early (single `exports` or `canonical` module) and update callers to reduce ambiguity.
- Always run `graphify query`, `graphify path`, or `graphify explain` before source-file deep dives on architecture tasks, and keep graph updates AST-only.
- Document Graphify best-practices and token-reduction tips in repository docs (see `docs/GRAPHIFY_TOKEN_TIPS.md`).

## CLI Command Catalog (existing repo commands)

Use `uv` by default for environment, package, script, and test commands.

### Environment and packages

```bash
uv venv create .venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
uv pip add spacy sentence-transformers
uv run python3 -m spacy download sv_core_news_sm
```

### Graphify

```bash
graphify update .
graphify query "How do speech classifications flow into manuscript figures?"
graphify explain "scripts/classify_speeches_parquet.py"
graphify path "classify_speeches_parquet.py" "render_manuscript_jinja.py"
```

### Core pipeline (legacy SQLite paths still documented in repo)

```bash
python scripts/ingest.py --sample
python scripts/classify.py --db data/swedish_parliament.db
python scripts/visualize.py --db data/swedish_parliament.db --out figures
python scripts/precompute_category_embeddings.py
python scripts/train_supervised.py
uvicorn web.app:app --reload
```

### Speech/parquet workflows

```bash
uv run python scripts/classify_speeches.py --db data/swedish_parliament.db
uv run python3 scripts/classify_speeches_parquet.py --input-dir data/speeches/parquet --out data/parquet/speech_classifications.parquet
CLASSIFIER_CPU_FRACTION=0.25 uv run python3 scripts/classify_speeches_parquet.py --sleep-every 50 --sleep-seconds 0.2
uv run python scripts/extract_speeches.py --src data/bulk_datasets --out data/speeches/parquet
uv run python scripts/extract_votering.py --src data/votering --out data/votering/parquet
uv run python scripts/extract_betankande.py --src data/bulk_datasets --out data/betankande/parquet
```

### Download/import/backfill utilities

```bash
uv run python scripts/download_speeches.py
uv run python scripts/download_votering.py
uv run python scripts/download_betankande.py
uv run python3 scripts/import_bulk_datasets.py --db data/swedish_parliament.db --zips data/bulk_datasets
uv run python3 scripts/backfill_text.py --db data/swedish_parliament.db --min-len 50
poetry run python scripts/backfill_party.py --db data/swedish_parliament.db
poetry run python scripts/clean_party_labels.py --db data/swedish_parliament.db
uv run python scripts/fix_dates.py --db data/swedish_parliament.db
```

### Training/evaluation/calibration

```bash
uv run python3 scripts/train_models.py --db data/swedish_parliament.db
uv run python scripts/train_hybrid_ensemble.py --db data/swedish_parliament.db --bert-model models/transformer_ideology_classifier/final
uv run python scripts/train_mlp_meta_learner.py --db data/swedish_parliament.db --epochs 50 --hidden 256 128
uv run python scripts/finetune_transformer_classifier.py --db data/swedish_parliament.db --epochs 3 --batch-size 8
poetry run python3 scripts/finetune_swedish_bert.py --db data/swedish_parliament.db
poetry run python3 scripts/finetune_swedish_bert.py --db data/swedish_parliament.db --retrain-ensemble
uv run python3 scripts/evaluate_ensemble.py --db data/swedish_parliament.db
uv run python3 scripts/evaluate_speech_gold_labels.py
uv run python3 scripts/run_calibration_checks.py
uv run python3 scripts/benchmark_embedding_models.py --db data/swedish_parliament.db
```

### Label generation, augmentation, and rhetoric

```bash
uv run python3 scripts/generate_gold_labels.py --db data/swedish_parliament.db --sample 2000
uv run python scripts/generate_speech_gold_labels.py --db data/swedish_parliament.db --target-per-category 30
uv run python3 scripts/augment_gold_labels.py --db data/swedish_parliament.db
uv run python3 scripts/augment_speech_gold_labels.py --db data/swedish_parliament.db
uv run python3 scripts/augment_rare_classes.py --db data/swedish_parliament.db --target-per-class 150
uv run python3 scripts/generate_rhetoric_sample.py --parquet-dir data/parquet --out data/parquet/speech_rhetoric_labels_sample.parquet --n 10
uv run python3 scripts/generate_rhetoric_zs_sample.py --parquet-dir data/parquet --out data/parquet/speech_rhetoric_labels_zs_sample.parquet --n 100
uv run python3 scripts/label_rhetoric_zero_shot.py --in logs/selected_10_speeches_full.csv --out logs/rhetoric_llm_labels.csv
uv run python3 scripts/ingest_rhetoric_labels.py --csv logs/rhetoric_llm_labels_selected_10.csv --db data/swedish_parliament.db
uv run python3 scripts/merge_rhetoric_into_parquet.py --parquet-dir data/parquet --out data/parquet/augmented_speech_gold_labels.parquet
```

### Active learning, self-training, and review

```bash
uv run python scripts/active_learning.py --db data/swedish_parliament.db --select 500 --label-model qwen2.5-coder-14b-32k:latest
uv run python3 scripts/export_active_learning_candidates.py --help
poetry run python3 scripts/self_train.py --db data/swedish_parliament.db --confidence 0.75 --max-per-class 500
uv run python3 scripts/self_train_speech.py --db data/swedish_parliament.db --confidence 0.90 --max-per-class 500 --retrain
uv run python3 scripts/ollama_review_gold_speeches.py --db data/swedish_parliament.db --model llama3.1:8b
uv run python3 scripts/ollama_review_from_stratified_report.py --report stratified_classification_report.md --model llama3.1:8b
```

### Figures and reporting

```bash
uv run python scripts/generate_figures.py --db data/swedish_parliament.db --out-dir figures/manuscript
uv run python scripts/visualize_voting.py --votering-parquet data/votering/parquet --out figures/voting
uv run python scripts/regenerate_all_visualizations.py
```

### Parquet and artifact conversion utilities

```bash
python scripts/export_tables_to_parquet.py --db data/swedish_parliament.db --out data/parquet --tables normalized_motions motions
python scripts/verify_parquet_exports.py --db data/swedish_parliament.db
python scripts/convert_artifacts_to_zstd.py --root data --ext .pkl .json --yes
```

### Lookup/debug helpers

```bash
python3 scripts/lookup_speeches.py --speech-id <id> --out logs/sample.csv
python3 scripts/lookup_speeches.py --speaker "Åkesson" --limit 50
python3 scripts/lookup_motions.py --motion-id <id>
python3 scripts/lookup_motions.py --party M --limit 50
python3 scripts/lookup_motions.py --author "Andersson"
```

### Test command

```bash
uv run pytest -q
```

