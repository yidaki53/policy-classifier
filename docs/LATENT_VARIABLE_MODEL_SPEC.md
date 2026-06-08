---
_agent_frontmatter:
  id: "docs/LATENT_VARIABLE_MODEL_SPEC"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

# Latent Variable Model Specification (Party Ideology)

## Purpose

Specify a concrete latent-variable formulation so party ideology is modeled as an unobserved construct measured through multiple noisy channels, rather than as a direct output of any single classifier.

## Measurement Frame

Latent quantity:

- Party-year latent ideology: theta_{p,t}

Observed indicators (channel-level summaries):

- Speech channel indicator: y_speech_{p,t}
- Motion channel indicator: y_motion_{p,t}
- Action channel indicator: y_action_{p,t}
- Optional fulfillment/contradiction indicator: y_fulfill_{p,t}

Baseline linear measurement equations:

- y_k_{p,t} = alpha_k + lambda_k * theta_{p,t} + epsilon_k_{p,t}

where k indexes channels and epsilon_k captures channel-specific noise.

## Identification and Constraints

- Fix one loading (for example lambda_action = 1) to set scale.
- Constrain signs so larger theta consistently maps to the same ideology direction.
- Estimate channel-specific residual variances to capture unequal noise across channels.

## Dynamic Structure (optional phase 2)

- theta_{p,t} = phi * theta_{p,t-1} + eta_{p,t}

This separates persistent ideological position from year-level shocks.

## Link-Confidence Integration

Use linkage confidence strata from:

- output/analysis/speech_action_link_confidence_strata.parquet

as either:

- observation weights in the action-linked indicators, or
- separate measurement-error components by stratum.

Minimum reporting requirement:

- report latent estimates with and without low-confidence strata and quantify delta.

## Validation Plan

1. Convergent validity:
   - correlate latent theta_{p,t} aggregates with external benchmark scores when available.
2. Internal robustness:
   - re-estimate under model-family and linkage-setting perturbations.
3. Uncertainty:
   - provide interval estimates for party-level latent scores and pairwise differences.

## Deliverables

- Model input table: party-year channel indicators with uncertainty metadata.
- Latent output table: party-year latent mean and interval bounds.
- Sensitivity table: latent score drift across linkage strata and model variants.
- Manuscript update: replace direct-score language with latent-estimate-under-assumptions language.

## Practical Sequencing in This Repo

1. Keep current deterministic and ensemble outputs as indicator generators.
2. Run link-confidence and uncertainty scripts before latent fitting.
3. Fit latent model on frozen artifact snapshot.
4. Validate against external benchmarks or explicitly document benchmark gap.
