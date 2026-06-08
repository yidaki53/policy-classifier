---
_agent_frontmatter:
  id: "manuscript/review/reviewer2_critique"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Reviewer #2 Critique (Major Revision)

## Overall recommendation

Major revision. The manuscript addresses an important and policy-relevant problem with a strong reproducibility orientation, but current framing and inferential discipline need tightening before publication.

## Major concerns

1. Scope-question mismatch
- The manuscript title and research question emphasize deterministic motion classification, while the reported pipeline and findings are multimodal (motions, speeches, votes) with ensemble methods and fairness-constrained linkage.
- This creates ambiguity about the true contribution and evaluation target.

2. Inferential risk from model quality vs downstream interpretation
- Reported speech baseline accuracy is low, yet downstream trend and consistency interpretations are prominent.
- The manuscript must explicitly constrain claims to descriptive/measurement outputs and distinguish calibration gains from classification validity.

3. Methods/results organization is difficult to audit for readers
- Results combine model inventory, data engineering, optimization, and interpretation in a single long section.
- Readers need clearer separation of: (a) what was estimated, (b) what diagnostics were passed, and (c) which conclusions are robust.

4. Limitations and non-causal boundaries are underdeveloped
- The current significance text notes sensitivity checks but does not explicitly articulate non-causal interpretation boundaries, possible linkage error propagation, and uncertainty due to dictionary/model choices.

5. Literature grounding and citation handling
- Core text-as-data and parliamentary-positioning literature should be explicitly cited in the question/method framing.
- Abstract should avoid citations per target-journal norms.

6. Data availability could be more submission-ready
- Current statement is good operationally, but publication submission will likely require explicit repository/archival identifiers and a concise statement language that clearly maps to journal policy.

## Minor concerns

1. Terminology consistency
- Terms such as ideology index, consistency, fulfillment, and contradiction should be explicitly defined in one concise paragraph.

2. Readability
- Several long paragraphs in Results could be split into compact subsections for faster reviewer verification.

3. Reproducibility cueing
- A concise list of “hard constraints on interpretation” would improve transparency.

## Remediation status (2026-05-31)

### Completed in current manuscript build

1. Scope and framing alignment
- Main narrative now consistently frames the contribution as multimodal and deterministic-first with bounded interpretation.

2. Inferential boundaries
- Results and conclusion explicitly separate descriptive inference from causal claims and distinguish model quality from claim strength.

3. Readability and terminology
- First-mention definitions were added for key constructs (ideology index, consistency, fulfillment, contradiction), with additional reader-facing prose and punctuation simplification.
- A compact "How to read the metrics" mini-guide is now included in the appendix.

4. Figure organization
- Intermediate/process figures were moved to the appendix; main Results now focuses on headline figures.

5. Provenance and reproducibility cues
- Figure captions include update timestamps and artifact-oriented provenance language.

### Final visual QA findings (resolved)

1. Appendix figure content policy mismatch (resolved)
- Figure 8 (Three-way Divergence) is now regenerated under the same excluded-party policy used for overlay outputs (`Unknown`, `Moderaterna`, `Vänsterpartiet`, `X`).

2. Data availability submission readiness (resolved)
- Placeholder language was replaced with a concrete public-access statement including repository location and release/commit citation expectations.

### Recommendation before submission

- Keep the current excluded-party policy synchronized across all manuscript figures if any figure-regeneration scripts are rerun.
- At production handoff, include the exact release tag and commit hash cited in the Data Availability statement.
