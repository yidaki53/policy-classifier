---
_agent_frontmatter:
  id: "docs/DEFINITION_OF_DONE"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Definition of Done — PLOS ONE Submission Readiness

Last updated: 2026-05-29

This document defines the criteria that must be satisfied before the manuscript is considered ready for PLOS ONE submission.

---

## 1. Jinja / Rendering

- [ ] All `{{ block_name }}` template variables resolve without errors (`make render` exits 0).
- [ ] No unresolved `{{ ... }}` placeholders remain in `manuscript/build/rendered_sections/`.
- [ ] `manuscript/build/manuscript_context.json` exists and is populated with live artifact values.
- [ ] `make combined` produces a valid `manuscript.md` from rendered sections.

**Status: ✅ DONE** (as of 2026-05-29T20:27:49Z)

---

## 2. Section Completeness

| Section | Status | Notes |
|---------|--------|-------|
| `01_title.md` | ✅ Done | Static, finalized |
| `01a_abstract.md` | ✅ Done | ≤300 words (195), 3-paragraph structure, metrics injected |
| `02_question.md` | ✅ Done | H1–H3 hypotheses, lit framing, bibliography seed |
| `03_results.md` | ⚠️ Needs work | Methods inventory complete; needs explicit H1/H2/H3 evaluation and coverage explanation |
| `04_significance.md` | ⚠️ Needs work | Limitations good; needs explicit H1/H2/H3 discussion linkage; add acknowledgments note |
| `05_data_availability.md` | ✅ Done | Parquet paths, scripts, context JSON cited |
| `06_acknowledgments.md` | ❌ Missing | PLOS ONE requires Acknowledgments section |

---

## 3. PLOS ONE Structural Requirements

PLOS ONE expects: Title page · Abstract · Introduction · Methods · Results/Discussion · Acknowledgments · References

**TQRS → PLOS ONE mapping:**

| TQRS Section | PLOS ONE Role |
|---|---|
| `01a_abstract.md` | Abstract |
| `02_question.md` | Introduction |
| `03_results.md` (methods inventory portion) | Methods |
| `03_results.md` (findings portion) | Results/Discussion |
| `04_significance.md` | Discussion (additional) |
| `05_data_availability.md` | Data Availability Statement |
| `06_acknowledgments.md` (to create) | Acknowledgments |

- [ ] Acknowledgments section created (`manuscript/sections/06_acknowledgments.md`).
- [ ] Section 03 narration explicitly evaluates H1, H2, H3 (one paragraph each).
- [ ] Section 04 references H1/H2/H3 in the limitations discussion.
- [ ] Continuous line numbers enabled in final submission file.

---

## 4. Rigor Checks

- [ ] **Low classifier accuracy contextualized**: Baseline speech accuracy `0.2033` is still well below ceiling (13 classes → chance ≈ 7.7%). Isotonic recalibration raises top-1 accuracy to `0.3709` on the same held-out set, while temperature scaling leaves top-1 accuracy unchanged. Must be stated in-text with explicit chance-level comparison and downstream interpretation guard.
- [ ] **Coverage=1.0000 explained**: Full 141 605/141 605 linkage coverage results from rel_dok_id-to-betankande bridging with explicit fallback assignment. Must note what proportion have actual structural vs. fallback links.
- [ ] **Non-causal language throughout**: All party comparisons are descriptive; no causal language (e.g., "caused", "effect", "driven by") used without qualification.
- [ ] **Calibration framing correct**: NLL improvement (2.52→1.72) described only as probability-quality improvement, not as evidence of better class assignment.
- [ ] **Sensitivity acknowledged**: Results sensitive to dictionary version, linkage rebalance, window sizes; sensitivity checks noted.
- [ ] **Figures referenced by path have actual files**: Check all figure paths injected in Section 03 exist on disk.

---

## 5. References / Bibliography

- [ ] `manuscript/bibliography/references.bib` contains ≥10 relevant references.
- [ ] Vancouver/ICMJE numbered citation style (`.csl` file present at `manuscript/bibliography/vancouver.csl`).
- [ ] All in-text `[@citekey]` citations have matching `.bib` entries.
- [ ] Pandoc can process `make pdf` or `make docx` without unresolved citation warnings.

**Current count: 13 references** — meets threshold.

---

## 6. Figures

- [ ] All figures referenced in manuscript exist as files.
- [ ] Each figure has documented provenance: producing script + input/output paths + UTC timestamp.
- [ ] Figures are publication-quality (≥300 dpi or vector for print).
- [ ] Figure captions written in section 03 for each referenced figure.

---

## 7. Keywords

- [ ] 4–8 PLOS ONE keywords defined.
- [ ] Keywords cover: computational method, political domain, data type, geographic scope.

**Draft keywords (see [docs/KEYWORDS.md](KEYWORDS.md)):**
Swedish Riksdag, parliamentary text analysis, multimodal political measurement, party ideology estimation, speech-action consistency, deterministic classifier, NLP classification, political text data

---

## 8. Data Availability Statement

- [ ] `05_data_availability.md` states all input data sources (open Riksdag API).
- [ ] All analysis artifacts located under `data/parquet/` and `output/analysis/` are described.
- [ ] Code repository URL (GitHub) included or noted as planned.
- [ ] Software versions documented (Python, key package versions via `requirements.txt`).

**Status: ✅ Largely done** — needs GitHub URL when public repo is ready.

---

## 9. Compliance Checks

- [ ] Abstract ≤300 words: **195 words** ✅
- [ ] No text promises specific p-values, R², or significance levels for effects not yet formally tested.
- [ ] Funding sources listed (even if "no funding" — PLOS ONE requires a statement).
- [ ] Competing interests statement (PLOS ONE submission metadata, not main body).
- [ ] Author contribution statement (CRediT taxonomy, PLOS ONE submission metadata).

---

## 10. Build Verification

- [ ] `make render` exits 0.
- [ ] `make combined` produces `manuscript.md` with all sections concatenated.
- [ ] `make pdf` or `make docx` produces a readable document without citation errors.
- [ ] `uv run pytest -q` passes.

---

## Priority Action List (to reach Done)

1. **Create `06_acknowledgments.md`** with funding/no-funding statement and data access acknowledgment.
2. **Amend section 03** to explicitly link results to H1/H2/H3 with one paragraph each.
3. **Amend section 03** to explain coverage=1.0000 (fallback linkage explanation).
4. **Amend section 04** to cross-reference H1/H2/H3 in the limitations.
5. **Add keywords** to `docs/KEYWORDS.md` and to submission metadata.
6. **Run `make combined`** and verify the final `manuscript.md`.
7. **Run `make pdf`** to check pandoc/citation processing.
8. **Check all figure paths** referenced in section 03 exist on disk.
