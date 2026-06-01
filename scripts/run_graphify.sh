#!/usr/bin/env bash
set -euo pipefail

# Lightweight wrapper to run graphify with the recommended local Ollama coder model.
# Usage: scripts/run_graphify.sh [--token-budget N] [--include "pattern/**"]

MODEL="qwen2.5-coder-14b-32k"
CONCURRENCY=1
DEFAULT_TOKEN_BUDGET=20000
INCLUDE_PATTERN=""
OUTDIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --token-budget)
      DEFAULT_TOKEN_BUDGET="$2"; shift 2;;
    --include)
      INCLUDE_PATTERN="$2"; shift 2;;
    --model)
      MODEL="$2"; shift 2;;
    --out)
      OUTDIR="$2"; shift 2;;
    --help|-h)
      echo "Usage: $0 [--token-budget N] [--include \"pattern/**\"] [--model MODEL]"; exit 0;;
    *)
      echo "Unknown arg: $1"; exit 2;;
  esac
done

echo "Running graphify update . (AST-only)"
graphify update .

if [ -z "$OUTDIR" ]; then
  OUTDIR="graphify-runs/${MODEL}-${DEFAULT_TOKEN_BUDGET}"
fi
mkdir -p "$OUTDIR"

EXTRACT_CMD=(graphify extract . --backend ollama --model "$MODEL" --max-concurrency $CONCURRENCY --token-budget "$DEFAULT_TOKEN_BUDGET" --out "$OUTDIR")
if [ -n "$INCLUDE_PATTERN" ]; then
  EXTRACT_CMD+=(--include "$INCLUDE_PATTERN")
fi

echo "Running: ${EXTRACT_CMD[*]}"
"${EXTRACT_CMD[@]}"
