# Pandoc usage for repository documentation

Quick examples to convert Markdown docs to PDF, HTML, or DOCX for sharing.

Convert Markdown to PDF (requires a LaTeX engine):

```bash
pandoc -s docs/GRAPHIFY_TOKEN_TIPS.md -o docs/GRAPHIFY_TOKEN_TIPS.pdf --pdf-engine=xelatex
```

Convert to HTML:

```bash
pandoc -s docs/GRAPHIFY_TOKEN_TIPS.md -o docs/GRAPHIFY_TOKEN_TIPS.html
```

Combine multiple Markdown files into a single PDF:

```bash
pandoc -s docs/REPRODUCIBILITY.md docs/GRAPHIFY_TOKEN_TIPS.md -o docs/REPRODUCIBILITY_FULL.pdf --pdf-engine=xelatex
```

Tip: keep source docs concise before converting; use `docs/GRAPHIFY_TOKEN_TIPS.md` recommendations to create short summaries for semantic extraction.
