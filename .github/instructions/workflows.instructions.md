---
applyTo: ".github/workflows/**"
---

# Workflow Instructions

- Keep one workflow definition per file. Do not concatenate multiple top-level YAML documents into the same workflow file.
- Match repository policy: use `uv` for environment setup, installs, and test execution unless there is a documented reason not to.
- Prefer explicit validation steps over implicit shell state. Checkout, Python setup, dependency sync, smoke import, and tests should each be separate steps.
- Use current GitHub Actions major versions and keep permissions minimal.
- Add `concurrency` when duplicate runs on the same ref would waste CI time.
- Validate workflow edits by checking YAML diagnostics and then running the narrowest repo validation the workflow is meant to enforce.