---
section_id: "05_data_availability"
section_title: "Data Availability"
objective: "Provide a formal Data Availability statement aligned with PLOS ONE policy language."
required_inputs:
  - "Current data and output artifact locations"
  - "Current repository paths for scripts and reproducibility metadata"
required_outputs:
  - "Formal data availability statement ready for manuscript submission"
required_metrics:
  - "none"
required_figures_tables:
  - "none"
provenance_requirements:
  - "Statement must identify where underlying data and generated artifacts are accessible"
update_triggers:
  - "Changes in repository layout or data-sharing mechanism"
  - "Changes in public archive/deposit details"
owner: "manuscript-agent"
status: "active"
last_updated_utc: "2026-06-08T17:00:00Z"
---

# Data Availability

All data and metadata underlying the findings reported in this manuscript are available within the project repository and its reproducible artifact directories. Source parliamentary records are retrieved from official Swedish Parliament open-data endpoints and normalized into compressed parquet datasets. Derived analysis tables used for results and figures are available under the repository analysis outputs, and generated figure assets are available in the manuscript and figure directories.

All scripts required to reproduce ingest, classification, linkage, analysis, and figure generation are included in the repository and are executed in a pinned Python environment using uv. The exact rendering/build context for the manuscript is exported at build time, and journal-readiness checks are exported alongside the manuscript build artifacts.

No participant-level restricted data are introduced by this project; all primary inputs originate from publicly available parliamentary materials.

The full reproducible project is publicly accessible at `https://github.com/yidaki53/policy-classifier`. Submission and production versions should cite the exact release tag and commit hash used for manuscript generation.

Archival DOI for the submission snapshot (`submission-2026-06-06-r3`): `https://doi.org/10.5281/zenodo.20572644`.

For production handoff, the recommended archival path is to create a versioned release snapshot and archive it in a long-term repository service (for example, a Zenodo-linked GitHub release). This preserves the exact manuscript-state code and artifacts and provides a persistent accession identifier for citation without changing the underlying access pathway described above. The archived record should include the release tag, commit hash, artifact directory inventory, and manuscript build timestamp used in the submitted version.

Numeric reporting policy: analysis JSON artifacts preserve machine-precision float values (IEEE754). In manuscript prose and figure captions, percentages are rounded for readability (typically to one decimal place unless otherwise stated). Where rounded text differs from full-precision values, the full-precision artifact is the reproducible reference.
