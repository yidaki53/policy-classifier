This directory contains the manuscript source files following the TQRS structure (Title, Question, Results, Significance).

Use the Makefile to regenerate figures and assemble the manuscript:

```
cd manuscript
make figures
make manuscript
```

Edit `manuscript/sections/*.md` and keep each section focused to its TQRS role. See `TQRS_GUIDELINES.md` for details.
