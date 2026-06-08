---
_agent_frontmatter:
  id: ".github/directory_tree"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Repository directory tree — quick reference

This file is a concise map of the repository and recommended layout. Agents should consult
this file first to find code, data, manuscripts, and tests without scanning the full tree.

Current top-level layout
-----------------------
- `__init__.py` — package metadata for `swedish_parliament_policy_classifier` (current package root)
- `manuscript.md` — project manuscript overview
- `README.md`, `requirements.txt` — basic usage & dependencies
- `analysis/` — analysis helper scripts (small utilities)
- `classifier/` — deterministic scorer, persistence helpers
- `db/` — sqlite schema and DB reader helpers
- `definitions/` — canonical category definitions (political_spectrum.yaml)
- `fetch/` — Riksdag API client and fetch helpers
- `figures/` — generated figures (do not edit)
- `manuscript/` — modular manuscript sections and build targets (Makefile)
- `models/` — trained model artifacts (joblib), model-related helpers
- `nlp/` — NLP helpers (embedding matcher, preprocess)
- `scripts/` — CLI entrypoints: `ingest.py`, `sync.py`, `classify.py`, `visualize*.py`
- `tests/` — pytest test suite
- `visualization/` — plotting helpers used by scripts
- `web/` — small web app for interactive exploration

Recommended (target) layout — move code into `src/`
-------------------------------------------------
To prepare this project for packaging and clearer imports, move the runtime package into
`src/swedish_parliament_policy_classifier/` and keep top-level convenience scripts and data as-is.

- `src/swedish_parliament_policy_classifier/`
  - `__init__.py` (package root)
  - `classifier/` (scorer, persist)
  - `db/` (schema, readers)
  - `definitions/` (political_spectrum.yaml + loader)
  - `fetch/` (Riksdag client)
  - `nlp/` (embedding_matcher, preprocess)
  - `analysis/` (aggregate)
  - `visualization/` (plotting helpers)
  - `web/` (Flask app code)
  - `models/` (small helpers; remove large binary artifacts to `models/` or `data/models/`)

- Top-level (unchanged)
  - `scripts/` — CLI entrypoints (keep here for easy invocation)
  - `data/` — database file(s), generated artifacts (do not commit)*
  - `figures/`, `output/` — generated outputs (do not edit)
  - `manuscript/` — manuscript sources (one file per section)
  - `tests/` — tests (update imports if package moves)

Notes & conventions
-------------------
- After moving to `src/`, add a `pyproject.toml` / `setup.cfg` to support `pip install -e .`.
- Keep `definitions/political_spectrum.yaml` immutable for a release; update via `definitions/loader.py`.
- Put large binary artifacts (trained models) in `models/` but consider storing them in `data/models/`
  if they are large and not version-controlled.
- Use `scripts/verify.py` (recommended) in CI to run a deterministic pipeline and assert checksums.

This file is intended to be the single quick reference for agents and contributors. If you restructure
the repository, update this file and `.github/copilot-instructions.md` accordingly.
