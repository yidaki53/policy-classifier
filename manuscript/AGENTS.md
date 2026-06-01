# manuscript/AGENTS.md

Scope: `manuscript/` and all nested files.

Agent routing rule:
1. Use `manuscript-agent` for all manuscript-related work (editing, formatting, section assembly, provenance updates, and manuscript builds).
2. Agent definition file: `manuscript/manuscript-agent.md`.

Purpose:
- Keep manuscript text synchronized with the latest reproducible analysis outputs.
- Enforce metadata-rich section files so agents know what each section must contain.

Non-negotiable manuscript rules:
1. Continuously update methodology/results claims when new figures, tables, metrics, or pipeline outputs are generated.
2. Never leave stale numbers in text after re-running analysis scripts.
3. Every figure/table claim must include provenance: producing script, key inputs, output path, and UTC run timestamp.
4. Methodology descriptions must reflect the current implemented pipeline (models, linkages, weighting, and forecasting), not a simplified or historical version.

Frontmatter requirement (mandatory for every manuscript source file):
1. Every markdown file under `manuscript/sections/` must start with YAML frontmatter.
2. Frontmatter must be comprehensive and section-specific. At minimum include:
	- `section_id`
	- `section_title`
	- `tqrs_role` (research Topic, research Question, rationale, significance)
	- `objective`
	- `required_inputs`
	- `required_outputs`
	- `required_metrics`
	- `required_figures_tables`
	- `provenance_requirements`
	- `update_triggers`
	- `owner`
	- `status`
	- `last_updated_utc`
3. If frontmatter is missing or incomplete, agents must add/fix it before editing section body text.

Section update workflow:
1. Read latest outputs in `output/analysis/` and manuscript figures/tables before editing section prose.
2. Update section numbers/claims to match current artifacts.
3. Record any newly referenced artifacts and metrics in the section body and frontmatter metadata.
4. Keep wording evidence-based and avoid unsupported claims.

Quality bar:
1. Prefer precise `n=` counts and named metrics (accuracy, NLL, RMSE, etc.) over vague qualitative language.
2. State assumptions and known limitations whenever metrics may be sensitive to linkage coverage, time windowing, or model configuration.

CLI commands (run from `manuscript/` unless noted):
1. `make figures` : regenerates figure assets via ingest/classify/visualization scripts.
2. `make check-frontmatter` : validates required YAML frontmatter markers/fields per section file.
3. `make render` : runs Jinja render pass and writes rendered section files/context JSON.
4. `make combined` : assembles rendered sections into `build/combined.md` and syncs `../manuscript.md`.
5. `make journal-check` : runs journal profile requirement checks and writes readiness report JSON.
6. `make pdf` : builds `build/manuscript.pdf` (and syncs to `../manuscript.pdf` when pandoc is available).
7. `make docx` : builds `build/manuscript.docx`.
8. `make html` : builds `build/manuscript.html`.
9. `make manuscript` : convenience target for PDF manuscript build.
10. `make stats` : prints combined-manuscript word count.
11. `make all` : runs full workflow (`figures` + `manuscript`).
12. `make clean` : removes build artifacts (`build/`, `../manuscript.md`, `../manuscript.pdf`).

Direct equivalents used by Make targets:
1. `uv run python ../scripts/render_manuscript_jinja.py --repo-root .. --manuscript-dir manuscript --out-dir manuscript/build/rendered_sections --context-out manuscript/build/manuscript_context.json`
2. `uv run python ../scripts/check_journal_requirements.py --repo-root .. --manuscript-dir manuscript --journal-profile manuscript/journal_profiles/plos_one.yaml --out manuscript/build/journal_requirements_report.json`
