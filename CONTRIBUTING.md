## Contributing — How to run and test locally

This project uses `uv` for virtual environment and package management. The
recommended workflow is below.

Quick start

1. Create the venv:

```bash
uv venv create .venv
```

2. Install dependencies and the package in editable mode:

```bash
uv pip install -r requirements.txt
uv pip install -e .
```

3. Run a smoke import test (verifies package install):

```bash
uv run python3 -c "import swedish_parliament_policy_classifier, analysis, models; print('imports OK')"
```

4. Run tests:

```bash
uv run pytest -q
```

Running scripts

- Use `uv run python3` to run the repository scripts with the installed package.
  Example:

```bash
uv run python3 scripts/visualize.py
uv run python3 scripts/visualize_advanced.py
```

Notes
- Prefer the package-qualified imports (e.g. `from swedish_parliament_policy_classifier import ...`) rather than relying on `PYTHONPATH` hacks.
- If you need to install optional heavy dependencies (e.g. spaCy models), follow their install docs and use `uv pip add`.
