---
_agent_frontmatter:
  id: "docs/PANDOC_USAGE"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Pandoc Usage

This project uses Pandoc to build a manuscript PDF from Markdown sections.

## Inputs

- `manuscript/sections/01_title.md`
- `manuscript/sections/02_question.md`
- `manuscript/sections/03_results.md`
- `manuscript/sections/04_significance.md`

## Build

From repo root:

```bash
uv run make -C manuscript manuscript
```

This concatenates section files into `manuscript.md` and, if Pandoc is installed, writes `manuscript.pdf`.

## Verify Pandoc

```bash
pandoc --version
```

## Reproducibility Notes

For academic rigor, record this metadata each time manuscript output is regenerated:

- Producing command and script/Make target
- Source section paths and git commit hash
- Output files and paths
- UTC timestamp
- Any non-default flags/options
