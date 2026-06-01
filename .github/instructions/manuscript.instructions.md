---
applyTo: "manuscript/**,manuscript.md"
---

# Manuscript Instructions

- Treat `manuscript/sections/*.md` as the source of truth. Do not hand-edit `manuscript/build/`, `manuscript.pdf`, or other rendered outputs except by running the supported render pipeline.
- Follow the manuscript narrative structure from `manuscript/TQRS_GUIDELINES.md`: keep claims organized as topic framing, research question, evidence/results, and concluding significance.
- Hard rule: every substantive choice must be explicitly justified where it appears. This includes case selection (for example, why Sweden), data-source selection, methodological design, model-family selection, and benchmark/reference selection. Each choice must include the rationale and the key tradeoff it resolves.
- Every quantitative claim must be traceable to a current artifact, producing script, key input paths, output paths, and a UTC timestamp.
- Prefer updating the source sections and then regenerating with `make render`, `make combined`, `make pdf`, or `make journal-check` from `manuscript/` as needed.
- Keep claims conservative when classifier quality is low. Distinguish descriptive findings, calibration improvements, and causal claims explicitly.
- When results are refreshed, check for stale numbers across `01a_abstract.md`, `03_results.md`, and `04_significance.md` together so the narrative stays synchronized.

## Definition Of Done For Manuscript Edits

- Source sections updated in `manuscript/sections/` only.
- Quantitative claims reconciled across abstract, results, and conclusion.
- All substantive choices include explicit rationale and tradeoff language in the relevant sections.
- Run provenance block updated when metrics, counts, figures, or scripts change.
- `make render` succeeds and regenerated context reflects the latest artifacts.
- `make journal-check` is run when submission-facing content or structure changes.