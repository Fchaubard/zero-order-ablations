#!/usr/bin/env bash
# Pull results/*.json from the sweep node into ./results, regenerate the paper data and tables,
# print the plain-text grid and (optionally) rebuild the PDF.
#   NODE=root@host PORT=22 bash scripts/pull_results.sh [--pdf]
set -euo pipefail
cd "$(dirname "$0")/.."
NODE=${NODE:-root@173.33.226.238}
PORT=${PORT:-21501}
PY=${PYTHON:-.venv/bin/python}
mkdir -p results
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" "$NODE:/workspace/zero-order-ablations/results/" results/
$PY scripts/collect_results.py --results results --paper paper
$PY -m zoo.status --out results --grid | tail -n 20
if [ "${1:-}" = "--pdf" ]; then
  (cd paper && make > /tmp/paper_make.log 2>&1 && echo "paper built: $(pdfinfo main.pdf | grep Pages)")
fi
