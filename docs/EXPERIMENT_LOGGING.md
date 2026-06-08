---
_agent_frontmatter:
  id: "docs/EXPERIMENT_LOGGING"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Experiment Logging Template

Use this template for every training/evaluation/analysis run.

## Run Metadata

- Run name:
- UTC timestamp:
- Operator:
- Git commit:
- Script/entrypoint:
- Dataset inputs (paths + row counts):
- Output artifacts (paths):

## Configuration

- Model/version:
- Hyperparameters:
- CPU/GPU settings (including throttle):
- Random seed(s):

## Methodology

- Objective:
- Assumptions:
- Preprocessing choices:
- Class balancing/calibration choices:

## Results

- Primary metrics:
- Secondary metrics:
- Error analysis notes:

## Figure/Table Provenance

For each figure/table produced:

- Name:
- Producing script:
- Input files:
- Output path:
- UTC timestamp:

## MLflow Mapping

- Experiment name:
- Run ID:
- Tracking URI:
- Logged artifacts:

## Next Actions

- What changed since prior run:
- What to test next:
