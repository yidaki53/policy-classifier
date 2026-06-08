---
_agent_frontmatter:
  id: "docs/say_vs_do_execution_plan"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Say-vs-Do Contradiction: Execution Plan

Status date: 2026-05-29

## Objective

Execute the implementation specification in phased order, with the first hard milestone:

- Link every speech to either a motion target or a vote-context target.

## Phase Plan

1. Phase A: Full coverage linkage (blocking milestone)
- Add deterministic all-speeches linker with prioritized fallback hierarchy.
- Integrate into analysis suite with optional flag.
- Verify 100 percent speech coverage.

2. Phase B: Candidate graph + axis scoring
- Emit top-K semantic candidates for unresolved/low-confidence links.
- Add 7-axis ideology scoring strictly from canonical definitions loader.

3. Phase C: Contradiction edge scoring
- Add NLI/rule blended contradiction features for each edge.

4. Phase D: Probabilistic aggregation
- Compute expected contradiction/uphold by party/topic/year.

5. Phase E: Pipeline integration + manuscript metrics
- Extend promise fulfillment and consistency outputs.
- Add contradiction-aware figures/tables and provenance.

## Current Step Execution

In progress:

- Phase E integration validation.

Completed:

1. Phase A implementation and validation.
2. Full-coverage linkage artifact generated at `data/parquet/speech_action_links.parquet`.
3. Coverage validated at 100 percent (`n_linked == n_speeches`).
4. Phase B canonical 7-axis scoring artifact generated at `output/analysis/speech_action_axis_scores.parquet`.
5. Phase C contradiction edge scoring artifact generated at `output/analysis/speech_action_contradiction_edges.parquet`.
6. Phase D expected contradiction aggregation artifact generated at `output/analysis/speech_action_expected_contradiction_party_topic_year.parquet`.

Completion criteria for Phase A:

1. Output parquet exists for all speeches.
2. `unique_linked_speeches == unique_speeches_in_classification`.
3. Each row has `action_type` in `{motion, vote}`.
