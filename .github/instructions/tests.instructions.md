---
applyTo: "tests/**/*.py,tests/*.py"
---

# Test Instructions

- Keep tests narrow and behavior-scoped. Prefer helper-level or script-slice tests over broad end-to-end runs unless the change specifically requires integration coverage.
- Follow the existing `tests/conftest.py` import pattern: tests should work with the `src/` layout first and avoid depending on ad hoc environment state.
- Prefer temporary parquet fixtures and small in-memory DataFrames over large real-corpus inputs.
- When touching `scripts/` or classifier behavior, add or update a focused regression test in the same change when practical.
- Validate with the narrowest relevant `uv run pytest -q ...` target before widening to the full suite.
