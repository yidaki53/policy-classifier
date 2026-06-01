History cleanup guide — remove large blobs from repo history

Warning: Rewriting git history is destructive. Back up the repository (mirror clone)
and coordinate with collaborators before pushing rewritten history.

Overview
- Goal: remove or relocate multi-GB blobs (e.g., `data/swedish_parliament.db`,
  original `data/bertopic_model.pkl`, `data/motion_topics.json`) from all commits
  so the repo can be pushed to GitHub without exceeding object size limits.
- Two primary approaches:
  1. Permanently remove sensitive/large files from history with `git-filter-repo`.
  2. Move large files into Git LFS with `git lfs migrate import` (note: GitHub LFS
     may enforce per-object limits ~2GB; very large blobs may still be rejected).

Recommended safe flow
1) Back up current repository (mirror):

```bash
git clone --mirror . ../policy-classifier-mirror.git
```

2) Inspect large objects (optional):

```bash
# list top 50 largest blobs in history (requires git)
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | sed -n 's/^blob //p' \
  | sort -k2 -n -r \
  | head -n 50
```

3) Use `git-filter-repo` to remove paths permanently (example):

```bash
pip install git-filter-repo
cd ../policy-classifier-mirror.git
# remove specific files/paths from history
git filter-repo --invert-paths --paths data/swedish_parliament.db data/bertopic_model.pkl data/motion_topics.json

# push cleaned mirror back to origin (force)
git remote add cleaned <your-cleaned-remote-or-github-url>
git push cleaned --all --force
git push cleaned --tags --force
```

4) Alternative: move files to LFS (if each large file is <2GB and LFS is acceptable):

```bash
git lfs install --local
git lfs migrate import --include="data/*.db,data/*.pkl,data/*.joblib,data/*.zip"
# inspect and push (may still be rejected if objects exceed server limits)
git push origin main --force-with-lease
```

5) After rewrite: inform collaborators to re-clone or run the provided rebase helper.

Safe helper script
- See `scripts/history_cleanup.sh` for a non-destructive dry-run and a gated
  execution mode (requires `RUN_HISTORY_CLEANUP=1` to actually perform rewrites).

If you want, I can run the dry-run analysis to list the largest blobs, or prepare a
mirror and produce the exact `git filter-repo` command set tailored to this repo.
