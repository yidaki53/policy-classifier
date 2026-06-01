---
applyTo: "scripts/**/*.py,scripts/*.py"
---

# Script Instructions

- Treat `scripts/` as the primary operational surface for ingest, classification, analysis, and manuscript rendering. Prefer small changes to existing entrypoints over adding overlapping scripts.
- Preserve parquet-first behavior. Do not introduce new SQLite-backed analysis or evaluation paths.
- Keep thermal-safe defaults for long-running jobs: conservative CPU fraction, explicit pacing, and resumable outputs when feasible.
- When a script changes reported metrics, figures, or manuscript claims, update provenance and the downstream manuscript context rather than only changing the script.
- Use `uv run python ...` or `uv run pytest ...` for validation commands.
- For new analysis outputs, prefer auditable parquet artifacts and stable column names so downstream manuscript/render scripts remain reproducible.
