# Graphify: Token-usage and extraction best practices

Practical tips to reduce LLM token usage when running `graphify extract` over this repository.

- Use the semantic cache and incremental extraction: `graphify extract` respects a semantic cache. Only changed files are re-sent to the model when the cache is available.
- Limit scope with `--include` / `--exclude` or run `graphify extract` only on changed directories instead of the whole repo.
- Choose smaller local models (less tokens) or reduce `--token-budget` per run. Prefer local Ollama models with smaller contexts when possible.
- Set `--max-concurrency 1` to avoid parallel requests and reduce peak usage if local inference is rate-limited.
- Pre-filter large documents (images, long PDFs) and convert them to concise text summaries before extraction.
- Use `--only-ast` or `graphify update .` to refresh AST-based links without semantic calls.
- Increase cache hit-rate by committing a stable JSON `graphify-out/.semantic_cache` between runs when files are unchanged.
- If you only need a few edges, use `graphify path` or `graphify query` to run cheaper, focused queries rather than a full `extract`.

Example: run an incremental semantic extraction with a local coder-optimized model and limited tokens

```bash
graphify update .
# recommended local model: qwen2.5-coder-14b-32k (use --model to override)
graphify extract . --backend ollama --model qwen2.5-coder-14b-32k --max-concurrency 1 --token-budget 10000 --include "classifier/**"
```

Reusable script: `scripts/run_graphify.sh` in this repo wraps the recommended flags:

```bash
scripts/run_graphify.sh [--token-budget N] [--include "pattern/**"]
```

See `docs/PANDOC_USAGE.md` for converting long docs to concise summaries before extraction.

## Graphify: Best Practices & Common Use Cases

These recommendations are grounded in hands-on runs against this repository (local Ollama models) and are tailored to reproducible engineering workflows.

- Prefer `graphify update .` (AST-only) when you only need structural links; run `graphify extract` only when you need semantic edges.
- Use package-level canonical exports (single-file `src/.../exports.py`) and `if False:` import-hint blocks to create explicit, high-confidence AST/semantic anchors for commonly-used symbols (e.g. `load_definitions`, `CategoryDef`). These inert hints dramatically reduce [INFERRED] edges in practice.
- When using local LLMs (Ollama), set `--max-concurrency 1` to avoid parallel local inference overloads and use a smaller `--token-budget` to limit per-chunk complexity.
- Keep a `graphify-runs/` directory for reproducible run outputs. Name runs by model and token-budget (e.g. `graphify-runs/qwen2.5-coder-14b-32k-20000`) so you can diff `graphify-out/graph.json` files across runs.
- Use a regression-checker (see `tools/graphify_regression.py`) to detect inferred-edge regressions against a baseline. Record baselines in `graphify-out/BASELINE.md`.
- For focused investigations, use `graphify path`, `graphify explain`, and `graphify query` on specific symbols instead of a full `extract`.

### Triaging INFERRED Edges

- Use `graphify explain "symbol()" --graph graphify-out/graph.json` to inspect inbound/outbound edges and the `surprises` list in `.graphify_analysis.json` for high-degree hubs.
- If `graphify explain` shows `[INFERRED]` for edges you expect to be explicit, add a small inert `if False:` import hint at the call site or consolidate a canonical export path and re-run `graphify update .` then `graphify extract`.
- Keep triage artifacts in `graphify-runs/triage/` so you can audit why an inference was made.

### Model-sensitivity matrix (recommended experiment)

To measure how model and token-budget affect inferred edges:

1. Clear semantic cache: remove `graphify-out/cache/semantic` (or run in a clean workspace).
2. For each model and token budget, run:

```bash
graphify update .
graphify extract . --backend ollama --model <MODEL> --max-concurrency 1 --token-budget <TOKENS> --out graphify-runs/<MODEL>-<TOKENS>
```

3. Parse `graphify-out/graph.json` for `links[*].confidence == 'INFERRED'` and record counts in `graphify-runs/model_sensitivity_matrix.csv`.

This repo includes `scripts/run_graphify.sh` and `tools/graphify_regression.py` to automate these steps; tweak `--include` to scope runs.

### When web docs are sparse

During this update we attempted to locate an authoritative external "Graphify" documentation page. Public results were sparse or organization-specific; therefore these repository-specific practices are the recommended approach for reproducible, low-noise semantic extraction.

If you want, I can also create a short runnable checklist `docs/GRAPHIFY_CHECKLIST.md` that codifies the steps above.
