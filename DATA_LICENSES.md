Data provenance and licensing
----------------------------

This repository contains two broad categories of material:

- Code, scripts, and supporting project files (licensed under the MIT License in `LICENSE`).
- Data files and datasets stored under `data/` (separate provenance; see below).

Riksdag open data
------------------

Many of the datasets used by this project are derived from the Swedish Riksdag's
open-data APIs (https://data.riksdagen.se). According to Riksdagens information
about open data, these datasets "får användas fritt" (may be used freely), but
users must always cite Sveriges riksdag as the source. See
https://www.riksdagen.se/sv/dokument-och-lagar/riksdagens-oppna-data/ for details.

Included data files
-------------------

Files in `data/bulk_datasets/` (e.g. `mot-*.json.zip`, `prop-*.json.zip`) appear
to be exported motion/proposition archives derived from the Riksdag API. If you
intend to redistribute these archives (for example in a public release or
package), comply with the Riksdag's attribution requirement and verify that no
additional third-party rights apply.

Important note on repository license
------------------------------------

The MIT license in `LICENSE` applies to the code and original files created by
the repository authors. It does NOT automatically relicense third-party data
contained in `data/`. Treat the data as governed by the original source's
terms (Riksdag open-data terms for Riksdag-origin data). When in doubt, link to
the original dataset and include attribution.

Practical recommendations
-------------------------

- Keep large raw datasets out of the main Git history where possible. Prefer
  to store data externally (S3, Zenodo, an institutional repository) and include
  download scripts in `scripts/` that fetch data on demand.
- If you must keep large files in the repo, use Git LFS and document this in
  `README.md`.
- Add an explicit `DATA_LICENSES.md` entry for any additional datasets you add
  in the future (include direct links and the exact license text if available).
