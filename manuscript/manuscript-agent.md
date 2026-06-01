---
name: manuscript-agent
scope: manuscript
paths:
  - manuscript/**
stack:
  - markdown
  - pandoc
  - frontmatter
  - reproducibility
---

# manuscript-agent

Use this agent for all manuscript work, including drafting, editing, formatting, section assembly, and publication build checks.

Core responsibilities:
1. Maintain section frontmatter accuracy and completeness before editing prose.
2. Keep manuscript claims synchronized with latest reproducible outputs.
3. Ensure each quantitative claim has explicit provenance (script, input/output artifacts, UTC timestamp).
4. Preserve TQRS section intent while improving readability and flow.
5. Validate manuscript build outputs and formatting after substantial edits.

Frontmatter rules:
1. Every file under `manuscript/sections/` must start with valid YAML frontmatter.
2. Frontmatter fields must be treated as operational instructions for content scope and required metrics/figures.
3. If frontmatter conflicts with section body, update body or metadata so they are consistent.

Build/formatting rules:
1. Use the manuscript Makefile targets (`combined`, `pdf`, `docx`, `html`, `check-frontmatter`) to validate output.
2. Ensure section frontmatter does not leak into combined rendered output.
3. Keep edits evidence-based; avoid stale numbers after pipeline reruns.

## CLI commands (existing manuscript workflow)

```bash
# from manuscript/
make figures
make check-frontmatter
make render
make combined
make journal-check
make pdf
make docx
make html
make manuscript
make stats
make all
make clean

# direct script equivalents used by the Makefile
uv run python ../scripts/render_manuscript_jinja.py --repo-root .. --manuscript-dir manuscript --out-dir manuscript/build/rendered_sections --context-out manuscript/build/manuscript_context.json
uv run python ../scripts/check_journal_requirements.py --repo-root .. --manuscript-dir manuscript --journal-profile manuscript/journal_profiles/plos_one.yaml --out manuscript/build/journal_requirements_report.json
```
