---
_agent_frontmatter:
  id: "reviewer_checklist"
  purpose: "Pre-submission readiness checklist for reviewer handoff of the Swedish Parliament Policy Classifier manuscript"
  steward: "manuscript-agent"
  edit_policy: "manual"
  generator: "manuscript prep script (2026-06-24)"
  generated: "2026-06-24T19:00:00Z"
  version: "1.0"
---

# Pre-Submission Reviewer Readiness Checklist

## Manuscript Requirements

### Structure & Completeness
- [ ] Title present and aligned with current scope
- [ ] Abstract present (section 01a_abstract)
- [ ] Research question stated (section 02_question)
- [ ] Methodology documented (section 03_methodology) — 12-stage speech pipeline, deterministic-first design
- [ ] Results reported (section 03_results) — with n counts, metrics, figures
- [ ] Conclusion / Significance (section 04_significance)
- [ ] Data Availability statement (section 05_data_availability)
- [ ] Acknowledgments (section 06_acknowledgments)
- [ ] Appendix / supplementary (section 07_appendix)

### Frontmatter Compliance
- [ ] All 9 section files have valid YAML frontmatter
- [ ] Each section has: section_id, section_title, objective, required_inputs, required_outputs, status, last_updated_utc
- [ ] All statuses are "active" (not "draft" or "stale")
- [ ] last_updated_utc within last 3 weeks

## Ethics & Legal
- [ ] Data license compliance: Swedish Riksdag open data — CC0 / public sector information
- [ ] No human subjects / no ethics approval required (public parliamentary records only)
- [ ] Competing interests statement
- [ ] Author contributions section
- [ ] Funding disclosure

## Data & Code Reproducibility
- [ ] Source data: Riksdag open data endpoints documented
- [ ] Code repository: github.com/yidaki53/policy-classifier (public)
- [ ] Environment: uv + Python 3.12, pinned dependencies via uv.lock
- [ ] Entry point: `make incremental-update` runs full pipeline
- [ ] Parquet artifacts committed or reproducible from raw data
- [ ] Zenodo archival DOI: 10.5281/zenodo.20572644
- [ ] All figures reproducible via `make figures` (under manuscript/)
- [ ] Journal-readiness check: `make journal-check` passes

## Figures & Tables
- [ ] All figures have provenance records (producing script, input/output paths, timestamp)
- [ ] Figure resolution >= 300 dpi
- [ ] All figures referenced in text use correct figure numbers
- [ ] Tables have consistent formatting
- [ ] Color-blind accessible palette used
- [ ] Captions are self-contained

## Bibliography
- [ ] 25+ entries (expanded from 15 to 25)
- [ ] Recent (2024-2026) literature cited for positioning
- [ ] Key missing papers covered:
  - [ ] Pätz et al. (2025) — German parliamentary speech classification
  - [ ] Nikolaev et al. (2023) — Multilingual party positioning
  - [ ] Osnabrügge et al. (2023) — Speech-Action multimodality
  - [ ] Ebrecht et al. (2024) — Cross-national debate scaling
  - [ ] Jaursch et al. (2025) — Multimodal political communication
  - [ ] Carlson et al. (2024) — Swedish pledge fulfillment
  - [ ] Grech et al. (2025) — Fairness-aware linkage
- [ ] All citations cross-referenced in text
- [ ] BibTeX keys are consistent and unique

## Language & Readability
- [ ] Abstract under 300 words (PLOS ONE limit)
- [ ] Methods section uses plain language where possible
- [ ] All acronyms defined on first use
- [ ] Claims are bounded: "descriptive", "observational", "under current assumptions"
- [ ] No unsupported causal language
- [ ] Numeric precision: percentages rounded to 1 decimal; full precision in JSON artifacts

## Pipeline Health (Post-Run)
- [ ] `make incremental-update` completes without errors
- [ ] Classification timeout fix applied (--speech-timeout 300)
- [ ] All downstream steps (profiles, linkage, contradictions, figures) succeeded
- [ ] `make manuscript` produces clean PDF
- [ ] `combined.md` renders with correct figure/table references

## Final Technical Checks
- [ ] `manuscript.md` synced from build/combined.md
- [ ] `manuscript.pdf` builds with pandoc
- [ ] `uv run pytest -q` passes (current: 53/53)
- [ ] Journal requirements report: `manuscript/build/journal_requirements_report.json` shows "ready"
- [ ] No TODOs or placeholder text remain in section files