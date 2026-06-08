---
_agent_frontmatter:
  id: "DATA_EXPORT_AND_COMMIT"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

Short guide: export DB tables to Parquet, compress artifacts, and prepare for commit/push

Prerequisites
- Activate the project's venv and install requirements:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

1) Export SQLite tables to Parquet

Export all tables (default) into `data/parquet/` (chunked):

```bash
python scripts/export_tables_to_parquet.py --db data/swedish_parliament.db --out data/parquet
```

Or export specific tables (recommended first):

```bash
python scripts/export_tables_to_parquet.py --db data/swedish_parliament.db --out data/parquet --tables normalized_motions motions classifications
```

2) Compress .pkl and .json artifacts to zstd siblings

This produces `file.pkl.zst` and `file.json.zst` without deleting the originals.

```bash
python scripts/convert_artifacts_to_zstd.py --root data --ext .pkl .json --yes
```

3) Verify outputs

- Parquet files should be in `data/parquet/` (open a sample with `python -c "import pandas as pd; print(pd.read_parquet('data/parquet/normalized_motions.parquet').shape)"`).
- Compressed artifacts are `*.zst` siblings (e.g. `data/motion_topics.json.zst`).

4) Prepare repository for push

- Update `.gitignore` to keep generated artifacts out of commits (already added: `.zst`, `data/parquet/`).
- If your local git history contains multi-GB objects, rewrite history before pushing (example using `git filter-repo` or `git lfs migrate import`). Only do this after confirming backups and coordinating with collaborators.

Example (filter-repo, *dangerous* — ensure backups):

```bash
# Install filter-repo if needed
pip install git-filter-repo

# Rewrite history to remove a path from all commits (example removing raw DB)
git clone --mirror . ../repo-mirror.git
cd ../repo-mirror.git
git filter-repo --invert-paths --paths data/swedish_parliament.db
cd ../policy-classifier
git remote add cleaned ../repo-mirror.git
git fetch cleaned
git checkout main
git reset --hard cleaned/main
```

Alternative: use `git lfs migrate import --include="*.db,*.pkl,*.joblib"` to move large objects to LFS (note: GitHub LFS may impose per-object limits).

5) Commit and push

Set commit identity as required (example):

```bash
git config user.name "yidaki53"
git config user.email "yidaki53@hotmail.com"
git add -A
git commit -m "chore(storage): export large tables to parquet; compress artifacts to zstd"
# If you rewrote history, use force push (coordinate first):
git push origin main --force-with-lease
```

Questions? Want me to run any step for you now (I can run export and compression locally)?
