#!/usr/bin/env bash
# Safe helper for preparing history cleanup. This script does NOT rewrite history
# unless RUN_HISTORY_CLEANUP=1 is set in the environment. It prints the recommended
# filter-repo commands and can run them when explicitly allowed.

set -euo pipefail

DRY_RUN=1
if [ "${RUN_HISTORY_CLEANUP:-0}" = "1" ]; then
  DRY_RUN=0
fi

echo "History cleanup helper"
echo "DRY_RUN=${DRY_RUN}"

MIRROR_DIR="../policy-classifier-mirror.git"

echo "1) Create a mirror backup (if not exists):"
echo "   git clone --mirror . ${MIRROR_DIR}"

echo "\n2) Example filter-repo command to remove large files from history:"
echo "   git filter-repo --invert-paths --paths data/swedish_parliament.db data/bertopic_model.pkl data/motion_topics.json"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "\nDry run mode: no destructive commands will be executed. To run, set RUN_HISTORY_CLEANUP=1 and re-run this script."
  exit 0
fi

echo "Running destructive cleanup now..."
if [ ! -d "${MIRROR_DIR}" ]; then
  git clone --mirror . "${MIRROR_DIR}"
fi

pushd "${MIRROR_DIR}"
pip install --user git-filter-repo || true
git filter-repo --invert-paths --paths data/swedish_parliament.db data/bertopic_model.pkl data/motion_topics.json
echo "Cleanup applied to mirror in ${MIRROR_DIR}. Inspect it before pushing." 
popd

echo "Done. If you want to push cleaned mirror to origin, add a remote and push with --force." 
