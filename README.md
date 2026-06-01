# policy-classifier

![CI](https://github.com/yidaki53/policy-classifier/actions/workflows/ci.yml/badge.svg)

Ingest Swedish Riksdag motions, classify them on a left–right ideological spectrum using deterministic YAML rules + semantic embeddings + a supervised fallback, and produce reproducible analyses and publication-quality figures. Includes an interactive web UI for human annotation.

## Features

- **SQLite-backed pipeline** — all motions, classifications, lineage, and annotations stored in a single file.
- **Three-stage classifier** — deterministic keyword rules → sentence-embedding similarity → TF-IDF + logistic regression fallback.
- **Active learning queue** — export low-confidence motions for efficient labeling.
- **Explainability** — per-motion JSON/HTML explanations identifying which stage fired and why.
- **Web annotation UI** — FastAPI + plain HTML; annotate motions at `http://localhost:8000`.
- **Publication figures** — party ideological profiles, heatmaps, and interactive Plotly charts.

## Installation (recommended)

This project uses `uv` for virtualenv and package management. `uv` ensures
commands run inside the project-managed environment and the recommended
developer workflow mirrors CI.

```bash
git clone https://github.com/yidaki53/policy-classifier.git
cd policy-classifier
uv venv create .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Optional heavy dependencies (safe fallbacks exist without them):

```bash
uv pip add spacy sentence-transformers
uv run python3 -m spacy download sv_core_news_sm   # Swedish lemmatiser
```

See [nlp/README.md](nlp/README.md) for fallback behaviour details.

## Quickstart

```bash
# 1. Ingest sample motions into SQLite
python scripts/ingest.py --sample

# 2. Classify motions
python scripts/classify.py --db data/swedish_parliament.db

# 3. Generate figures
python scripts/visualize.py --db data/swedish_parliament.db --out figures

# 4. (Optional) Pre-compute sentence embeddings for faster classification
python scripts/precompute_category_embeddings.py

# 5. (Optional) Train supervised fallback on labeled data
python scripts/train_supervised.py

# 6. (Optional) Start the annotation web UI
uvicorn web.app:app --reload   # then open http://localhost:8000

## Reproducibility & CLI

- Full reproduction steps and environment notes: [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
- Canonical CLI commands for zero-shot labeling and streaming runs: [docs/CLI_COMMANDS.md](docs/CLI_COMMANDS.md)

- Graphify token-reduction tips: [docs/GRAPHIFY_TOKEN_TIPS.md](docs/GRAPHIFY_TOKEN_TIPS.md)
- Pandoc usage for docs: [docs/PANDOC_USAGE.md](docs/PANDOC_USAGE.md)

```



## Manuscript PDF and Figures

This repository produces a manuscript PDF (`output/manuscript/manuscript.pdf`) and publication-ready figures as part of its reproducible pipeline. See `manuscript/README.md` and the new [README_MANUSCRIPT_NOTE.md](README_MANUSCRIPT_NOTE.md) for details on how the manuscript and figures are generated and how they relate to the codebase.

## Outputs

| Artifact | Description |
|---|---|
| `data/swedish_parliament.db` | SQLite DB (motions, classifications, annotations, lineage) |
| `figures/party_profiles.png/.pdf` | Stacked party-profile figure |
| `figures/party_profiles_final.png/.pdf` | Combined ideological placement + heatmap (publication-ready) |
| `data/active_learning_queue.csv` | Low-confidence motions for human review |
| `data/category_embeddings.pkl` | Pre-computed category embedding cache |

## Running tests

Run tests inside the project environment using `uv`:

```bash
uv pip install -r requirements.txt
uv run pytest -q
```

CI note: the project's GitHub Actions uses an editable install and a smoke
import before running the test suite: `uv pip install -e .` then a smoke
import and `uv run pytest -q`. Ensure an editable install succeeds locally
before opening a PR.

## Project layout

```
classifier/   deterministic scorer and persistence helpers
db/           SQLite schema and connection utilities
definitions/  YAML political-spectrum definitions (immutable source of truth)
fetch/        Riksdag API client
manuscript/   Chapter sources (Markdown) and build Makefile
nlp/          Text preprocessing, embeddings, and embedding cache
scripts/      Orchestration scripts (ingest, classify, visualize, explain, …)
tests/        Pytest test suite
visualization/ Figure generation modules
web/          FastAPI annotation UI + static HTML front-end
```

## Contributing

1. Fork the repository and create a feature branch.
2. Follow the deterministic-first principle: add or update YAML keyword rules in `definitions/political_spectrum.yaml` before reaching for ML.
3. All new behaviour must be covered by a test in `tests/`.
4. Run `pytest -q` and confirm it passes before opening a pull request.
5. Do not commit generated artifacts (`data/*.db`, `data/*.pkl`, `models/*.joblib`, `figures/`).

## Author

Robin Öberg — robinoberg@live.com

## License

- **Code:** MIT — see `LICENSE`.
- **Data:** Many datasets used by this project are derived from the Swedish
	Riksdag's open-data APIs (https://data.riksdagen.se). Riksdagens öppna data
	may be used freely but requires attribution to Sveriges riksdag. See
	`DATA_LICENSES.md` for provenance details and redistribution guidance.
