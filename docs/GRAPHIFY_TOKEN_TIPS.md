---
_agent_frontmatter:
  id: "docs/GRAPHIFY_TOKEN_TIPS"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Graphify Token Tips

Use Graphify before broad source-file exploration whenever the question is about structure, ownership, or relationships across modules.

## Fast Query Patterns

- `graphify query "How do speech classifications flow into manuscript figures?"`
- `graphify explain "scripts/classify_speeches_parquet.py"`
- `graphify path "classify_speeches_parquet.py" "render_manuscript_jinja.py"`

## When To Use What

- Use `graphify query` for broad architecture or navigation questions.
- Use `graphify explain` when one file, module, or concept is the anchor.
- Use `graphify path` when you need the dependency chain between two symbols or files.
- Only fall back to source-file reads when Graphify does not provide enough detail for an exact edit or line-level verification.

## Repo-Specific Guidance

- Keep Graphify in AST-only mode for this repository workflow.
- Run `graphify update .` after substantial refactors or after adding new scripts that should appear in the project graph.
- Prefer Graphify for manuscript provenance questions that cross scripts, outputs, and sections; it is cheaper than opening many files.
- When discussing architecture in chat, include the exact Graphify command you used so later agents can reproduce the same path or query.

## Suggested Prompts

- `graphify query "What scripts produce the artifacts cited in manuscript/sections/03_results.md?"`
- `graphify path "run_speech_analysis_suite.py" "analyze_consistency_trends.py"`
- `graphify explain "exports.py"`