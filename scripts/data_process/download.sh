#!/usr/bin/env bash
set -euo pipefail

# FineWeb-Edu sample/10BT
#
# Split:
#   train: 000_00000.parquet ... 012_00000.parquet
#   val:   013_00000.parquet
#
# wget -c resumes partially downloaded files.

BASE_URL="https://hf-mirror.com/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT"

TRAIN_DIR="data/fineweb_raw/train"
VAL_DIR="data/fineweb_raw/val"

mkdir -p "${TRAIN_DIR}" "${VAL_DIR}"

echo "===== Downloading training split: 000-012 ====="
wget -c \
  "${BASE_URL}/"{000..012}_00000.parquet \
  -P "${TRAIN_DIR}"

echo "===== Downloading validation split: 013 ====="
wget -c \
  "${BASE_URL}/013_00000.parquet" \
  -P "${VAL_DIR}"

echo "===== Download complete ====="