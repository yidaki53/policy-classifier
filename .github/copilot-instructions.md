---
_agent_frontmatter:
  id: ".github/copilot-instructions"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Copilot Instructions for swedish_parliament_policy_classifier

This repository ingests Swedish Riksdag motions, classifies them deterministically, and produces reproducible analyses and publication-ready figures. Follow these instructions when using Copilot/agents on this project.

Project Ground Truth
- Primary ingest entrypoint: `scripts/ingest.py` (sample & dev)
- Incremental sync: `scripts/sync.py` (live Riksdag API)
- Classification: `scripts/classify.py` (deterministic scorer)
- Figure generation: `scripts/visualize.py`, `scripts/visualize_advanced.py`, `scripts/visualize_interactive.py`
- DB schema and lineage: `db/schema.py`
- Canonical definitions: `definitions/political_spectrum.yaml`
- Manuscript source (TQRS): `manuscript/` (see `manuscript/TQRS_GUIDELINES.md`)

Non-Negotiable Rules
1. Do not edit generated artifacts in `data/`, `figures/`, or `output/`. Regenerate using the scripts.
2. Pydantic models (`models/models.py`) are the source-of-truth for in-repo data shapes.
3. Deterministic rules live in `definitions/political_spectrum.yaml`. Changes must be versioned and reviewed.
4. All code that transforms or normalizes motions must create lineage entries in `lineage`.
5. Manuscript drafts must follow the TQRS structure (Title, Question, Results, Significance). See `manuscript/TQRS_GUIDELINES.md`.
6. Academic rigor is mandatory: document methods, assumptions, parameter/model choices, and progress as you go; every reported figure/table must include reproducibility provenance (producing script, key inputs, outputs, and run timestamp).
7. Parquet-first execution is mandatory for analysis/training/evaluation: use compressed Parquet (`.parquet` with zstd) under `data/parquet/` and `output/`; treat SQLite (`data/swedish_parliament.db`) as legacy/migration-only.
8. Thermal-safe defaults are mandatory for long-running jobs: use conservative CPU fraction and pacing by default, and only increase resource usage with explicit user intent.

Default Task Workflow
1. Edit or add code under `scripts/`, `classifier/`, or `analysis/`.
2. Run `scripts/sync.py` (live) or `scripts/ingest.py` (sample) to populate `data/swedish_parliament.db`.
3. Run `scripts/classify.py` to produce `classifications` and party profiles.
4. Generate figures with `scripts/visualize*`.
5. Update manuscript sections in `manuscript/sections/` and build using `manuscript/Makefile`.
6. Run tests (`pytest`) and ensure deterministic behavior; include new tests for new logic.


## Package Management and Python Execution

- **MANDATORY:** Always use [`uv`](https://github.com/astral-sh/uv) for all package management (install, add, remove, update) and for running Python commands. This applies to all development, testing, and production scripts.
	- Create the venv: `uv venv create .venv`
	- Install dependencies: `uv pip install -r requirements.txt`
	- Add/remove packages: `uv pip add <package>` / `uv pip remove <package>`
	- Run Python scripts and tests: `uv run python ...` or `uv run pytest ...`
- Do **not** use `pip`, `python -m pip`, or `python` directly except in rare, explicitly documented cases (e.g., bootstrapping uv itself).

	- CI note: The repository's CI performs an editable install and a smoke-import before running tests. Locally, verify `uv pip install -e .` succeeds and then run `uv run pytest -q`.

Environment & Setup
- Preferred: use `uv` to create the project venv (user policy). Example: `uv venv create .venv`.
- Fallback: `python3 -m venv .venv && . .venv/bin/activate` (only if uv is unavailable).
- Install dependencies: `uv pip install -r requirements.txt`.

When generating manuscript text or claims, always tie claims to evidence, cite figures/tables and link to the exact script that produced them.

## Graphify Context Policy

- Check `graphify-out/graph.json` first for architecture, lineage, or dependency questions. Read source files only when graph evidence is insufficient.
- Update the graph after substantial edits with `graphify update .`.
- Keep graph outputs excluded from git (already in `.gitignore`).
- Use explicit graph paths when referencing context: `graphify-out/graph.json` and `graphify-out/GRAPH_REPORT.md`.
- If graphify output is missing, run `graphify update .` before cross-repo or architecture questions.
- Use Graphify in AST-only mode for this repository workflow. Do not use semantic extraction.

### Quick efficiency protocol (token-saving default)

Follow this sequence unless the task is trivial:

1. Graphify first (`query`, `explain`, or `path`).
2. Open only files needed for the concrete edit.
3. Keep generated YAML/Markdown machine-readable via valid frontmatter.
4. Validate with the narrowest command first.

Preferred Graphify command map:

- Architecture/question answering: `graphify query "<question>"`
- One module/symbol: `graphify explain "<target>"`
- Relationship tracing: `graphify path "<A>" "<B>"`

Fallback rule:

- Use broad repo search only when graph output is incomplete for an exact code change.

## Frontmatter Policy (YAML and Markdown)

- Markdown files should begin with YAML frontmatter (`---` block at top of file).
- Generated Markdown and YAML artifacts should include `_agent_frontmatter` for automation and routing.
- Frontmatter MUST be valid YAML 1.2: spaces only (no tab indentation), stable keys, simple scalar values where possible.
- Keep frontmatter minimal and high-signal; avoid large prose blocks in metadata.

Recommended fields:

- `id`: stable logical identifier.
- `purpose`: short machine/human summary.
- `steward`: owning area.
- `edit_policy`: `manual` or `generated_do_not_edit`.
- `generator`: source script/module for generated files.

Suggested markdown frontmatter template:

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

## CUDA & NVSHMEM (GPU) notes

- **Problem:** A CUDA-enabled PyTorch wheel can fail to import with an error like "ImportError: libnvshmem_host.so.3: cannot open shared object file". This means the system NVSHMEM runtime required by the CUDA build is missing or not on the dynamic loader path.
- **Fixes (choose one):**
	- **System install (recommended):** Install the NVSHMEM runtime matching your CUDA version and refresh the linker cache. Example (Debian/Ubuntu, CUDA 13):
		- `sudo apt-get update && sudo apt-get install -y libnvshmem3-cuda-13 && sudo ldconfig`
	- **Temporary loader path (no sudo):** Export the NVSHMEM library directory before running project commands (useful for one-off runs):
		- `export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13:$LD_LIBRARY_PATH`
		- Or one-off: `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/13 uv run python3 scripts/generate_rhetoric_zs_sample.py ...`
	- **CPU-only fallback:** If you cannot update system libs and do not need GPU, install a CPU-only PyTorch wheel via `uv add torch -U` or prefer PyTorch's prebuilt wheels:
		- `uv add --find-links https://download.pytorch.org/whl/torch_stable.html torch -U`
- **Verify:** After applying a fix, confirm with:
	- `uv run python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
- **Note:** Prefer running project commands with `uv run` so they execute inside the project-managed environment.

## graphify

For any question about this repo's architecture, structure, components, or how to add/modify/find
code, your first action should be `graphify query "<question>"` when `graphify-out/graph.json`
exists. Use `graphify path "<A>" "<B>"` for relationship questions and `graphify explain "<concept>"`
for focused-concept questions. These return a scoped subgraph, usually much smaller than the full
report or raw grep output.

Triggers: "how do I…", "where is…", "what does … do", "add/modify a <component>",
"explain the architecture", or anything that depends on how files or classes relate.

If `graphify-out/wiki/index.md` exists, use it for broad navigation. Read `graphify-out/GRAPH_REPORT.md`
only for broad architecture review or when query/path/explain do not surface enough context. Only read
source files when (a) modifying/debugging specific code, (b) the graph lacks the needed detail, or
(c) the graph is missing or stale.

Type `/graphify` in Copilot Chat to build or update the graph.

## Customization Layout

- Keep this file limited to repo-wide defaults that apply across the project.
- Put file-specific rules in `.github/instructions/*.instructions.md` so Copilot only loads them when the matching files are in scope.
- Use `AGENTS.md` for agent-facing workflow policy and high-level repo navigation. Keep it consistent with this file rather than duplicating large blocks verbatim.
- When working on manuscript files, prefer the manuscript-specific instructions under `.github/instructions/` over adding more manuscript detail here.
- When working on GitHub Actions or workflow YAML, prefer the workflow-specific instructions under `.github/instructions/` over adding CI-specific detail here.
