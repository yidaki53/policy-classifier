This directory contains the manuscript source files following the TQRS structure (Title, Question, Results, Significance).

Use the Makefile to regenerate figures and assemble the manuscript:

```
cd manuscript
make figures
make render
make journal-check
make manuscript
```

Edit `manuscript/sections/*.md` and keep each section focused to its TQRS role. See `TQRS_GUIDELINES.md` for details.

Publication workflow assets:

- Target journal profile: `manuscript/journal_profiles/plos_one.yaml`
- Seed bibliography: `manuscript/bibliography/references.bib`
- Rendered sections: `manuscript/build/rendered_sections/*.md`
- Render context/provenance: `manuscript/build/manuscript_context.json`
- Journal readiness report: `manuscript/build/journal_requirements_report.json`
