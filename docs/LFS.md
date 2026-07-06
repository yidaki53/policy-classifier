# Git LFS budget remediation

Symptom: CI `actions/checkout` fails with `This repository exceeded its LFS budget`.

Cause: GitHub is rejecting `git lfs fetch` for this repo. The tracked files listed by `git lfs ls-files --size` total well under 10 GB, so the failure is most likely a GitHub account-level LFS quota or billing issue (storage or bandwidth), not a repository-size problem.

Fix options:
- Increase/purchase LFS quota: open GitHub → Settings → Billing → Git LFS and add a data pack or enable billing.
- Purge large LFS-tracked files from history if you prefer a leaner repo (coordinate with collaborators before force-pushing):

```bash
git lfs uninstall
git filter-repo --path-glob 'data/bulk_datasets/*.zip' --path-glob 'data/**/*.parquet' --path-glob 'data/**/*.db' --path-glob 'output/**/*.parquet' --path-glob 'figures/**/*.parquet' --path-glob 'models/**/*.pkl.zst' --path-glob 'models/**/*.pt' --path-glob 'models/**/*.safetensors' --path-glob 'models/**/optimizer.pt' --path-glob 'models/**/tokenizer.json' --path-glob 'src/models/*.pkl' --invert-paths --force
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git add .gitattributes
git commit -m "chore: remove large LFS-tracked binaries from history"
git push origin main --force
```

Re-enable LFS afterward if desired:
```bash
git lfs install
git add .gitattributes
git commit -m "chore: restore LFS tracking rules"
git push origin main