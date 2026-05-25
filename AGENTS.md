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

- Focus on classifier pipeline, YAML definitions, Parquet-based data layer, and NLP preprocessing.
- Keep changes small and reversible. Prefer deterministic rules before ML.
- Preserve naming contracts and deterministic-first principle.
- Add or update tests in the same change as code.
- Validate with `uv run pytest -q` before submitting.
  - CI runs `uv pip install -e .`, a smoke import test, and `uv run pytest -q`.
- **For architecture questions, use the graphify skill (`/graphify`) FIRST.** Query `graphify-out/graph.json` via `graphify query`, `graphify explain`, or `graphify path` before reading source files directly. This saves tokens and provides cross-document context that file-by-file reading misses.
- Use relative paths only. Never use absolute paths.
- Update the graph after substantial edits with `graphify update .`.
- Use local Ollama semantic extraction only when Ollama and the model are present; otherwise use `graphify update` and warn that semantic extraction was skipped.

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

Recommendations for future agents:

- Prefer small, incremental changes and re-run `graphify update .` between edits to measure impact.
- Use `if False:` import hints near call-sites for high-confidence linking when refactors are costly.
- Consolidate canonical export paths early (single `exports` or `canonical` module) and update callers to reduce ambiguity.
- When running `graphify extract`, prefer smaller local models (or lower token budgets) and rely on semantic cache to avoid re-sending unchanged files.
- Document Graphify best-practices and token-reduction tips in repository docs (see `docs/GRAPHIFY_TOKEN_TIPS.md`).

