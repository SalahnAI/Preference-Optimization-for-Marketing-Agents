#!/usr/bin/env bash
# Stream the Criteo Display Advertising Challenge data and keep only the first N
# rows of train.txt (the only file with click labels).
#
# The full archive is 4.58 GB (train.txt ~11 GB uncompressed). We don't need it
# all to build a few hundred pseudo-campaigns, so we decompress on the fly and
# stop early: `head` closing the pipe sends SIGPIPE upstream and curl aborts.
# Archive member order is readme.txt -> test.txt -> train.txt, so tar reads past
# the first two (downloading ~0.5 GB) before train.txt starts streaming.
#
# Usage:  scripts/get_criteo.sh [N_ROWS]   (default 1,500,000)
#
# Source: figshare mirror of the official Criteo Kaggle DAC dataset.
set -euo pipefail

ROWS="${1:-1500000}"
URL="https://ndownloader.figshare.com/files/10082655"
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
OUT="$OUT_DIR/train.txt"
TMP="$OUT_DIR/.train.partial"

mkdir -p "$OUT_DIR"
echo "Streaming first $ROWS rows of train.txt from Criteo DAC (figshare)..."
# pipefail would flag SIGPIPE on curl/tar; that's expected, so guard the pipeline.
set +o pipefail
curl -fL --retry 3 --max-time 1800 "$URL" \
  | tar -xzO train.txt 2>/dev/null \
  | head -n "$ROWS" > "$TMP"
set -o pipefail

mv "$TMP" "$OUT"
echo "Wrote $(wc -l < "$OUT") rows -> $OUT  ($(du -h "$OUT" | cut -f1))"
echo "Now build campaigns:  PYTHONPATH=src python -m marketing_agent.data_prep --source criteo --n 400"
