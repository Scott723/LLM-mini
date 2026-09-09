#!/usr/bin/env bash
set -euo pipefail

# Convenient wrapper for FineWeb-Edu tokenization.
#
# Usage:
#   bash scripts/data_process/tokenize.sh train
#   bash scripts/data_process/tokenize.sh val
#   bash scripts/data_process/tokenize.sh all
#
# Optional overrides:
#   TOKENS_PER_SHARD=50000000
#   TOKENIZE_BATCH_SIZE=1024
#   PYTHON_BIN=python
#
# Important:
# tokenize_fineweb_auto_index.py only continues the OUTPUT FILE INDEX.
# It does not remember which Parquet files have already been processed.
# Re-running the same split will append duplicate tokenized data.

SCRIPT_DIR="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
)"

PROJECT_ROOT="$(
  cd "${SCRIPT_DIR}/../.." && pwd
)"

cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TOKENIZER="scripts/data_process/tokenize_fineweb_auto_index.py"

TOKENS_PER_SHARD="${TOKENS_PER_SHARD:-50000000}"
TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-1024}"

MODE="${1:-all}"

tokenize_train() {
  echo "============================================================"
  echo "[train] FineWeb-Edu 000-012"
  echo "============================================================"

  "${PYTHON_BIN}" "${TOKENIZER}" \
    --input-dir data/fineweb_raw/train \
    --output-dir data/fineweb_tokenized_v1/train \
    --pattern "*.parquet" \
    --tokens-per-shard "${TOKENS_PER_SHARD}" \
    --batch-size "${TOKENIZE_BATCH_SIZE}" \
    --prefix train
}

tokenize_val() {
  echo "============================================================"
  echo "[val] FineWeb-Edu 013"
  echo "============================================================"

  "${PYTHON_BIN}" "${TOKENIZER}" \
    --input-dir data/fineweb_raw/val \
    --output-dir data/fineweb_tokenized_v1/val \
    --pattern "*.parquet" \
    --tokens-per-shard "${TOKENS_PER_SHARD}" \
    --batch-size "${TOKENIZE_BATCH_SIZE}" \
    --prefix val
}

if [[ ! -f "${TOKENIZER}" ]]; then
  echo "ERROR: tokenizer not found: ${TOKENIZER}" >&2
  exit 1
fi

case "${MODE}" in
  train)
    tokenize_train
    ;;
  val)
    tokenize_val
    ;;
  all)
    tokenize_train
    echo
    tokenize_val
    ;;
  *)
    echo "Usage: bash scripts/data_process/tokenize.sh {train|val|all}" >&2
    exit 2
    ;;
esac

echo
echo "============================================================"
echo "Tokenization finished: ${MODE}"
echo "============================================================"
