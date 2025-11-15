#!/usr/bin/env bash
set -euo pipefail

# ============================
# Argument parsing
# ============================
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 MODEL.scad [OUTDIR]"
    exit 1
fi

MODEL="$1"
OUTDIR="${2:-renders}"

# ============================
# Config
# ============================
IMG_W=800
IMG_H=600
ELEV=30      # camera elevation
DIST=300     # camera distance

# OpenSCAD path (can override with env var)
OPENSCAD="${OPENSCAD:-openscad}"

mkdir -p "$OUTDIR"

echo "Rendering views for '$MODEL' into '$OUTDIR/'"
echo

# Track time
START_TIME=$(date +%s.%N)

# Track background PIDs
PIDS=()

echo "Scheduling 6 canonical views (front, back, left, right, top, bottom)..."
echo

# ============================
# Canonical views (parallel)
# ============================

# Front
"$OPENSCAD" \
  --viewall --autocenter \
  --camera=0,0,0,0,0,0,$DIST \
  --imgsize=${IMG_W},${IMG_H} \
  -o "$OUTDIR/view_front.png" \
  "$MODEL" &
PIDS+=($!)

# Back
"$OPENSCAD" \
  --viewall --autocenter \
  --camera=0,0,0,0,180,0,$DIST \
  --imgsize=${IMG_W},${IMG_H} \
  -o "$OUTDIR/view_back.png" \
  "$MODEL" &
PIDS+=($!)

# Right
"$OPENSCAD" \
  --viewall --autocenter \
  --camera=0,0,0,0,90,0,$DIST \
  --imgsize=${IMG_W},${IMG_H} \
  -o "$OUTDIR/view_right.png" \
  "$MODEL" &
PIDS+=($!)

# Left
"$OPENSCAD" \
  --viewall --autocenter \
  --camera=0,0,0,0,-90,0,$DIST \
  --imgsize=${IMG_W},${IMG_H} \
  -o "$OUTDIR/view_left.png" \
  "$MODEL" &
PIDS+=($!)

# Top
"$OPENSCAD" \
  --viewall --autocenter \
  --camera=0,0,0,90,0,0,$DIST \
  --imgsize=${IMG_W},${IMG_H} \
  -o "$OUTDIR/view_top.png" \
  "$MODEL" &
PIDS+=($!)

# Bottom
"$OPENSCAD" \
  --viewall --autocenter \
  --camera=0,0,0,-90,0,0,$DIST \
  --imgsize=${IMG_W},${IMG_H} \
  -o "$OUTDIR/view_bottom.png" \
  "$MODEL" &
PIDS+=($!)

echo
echo "Waiting for all renders to finish..."

for pid in "${PIDS[@]}"; do
  wait "$pid"
done

END_TIME=$(date +%s.%N)
TOTAL=$(echo "$END_TIME - $START_TIME" | bc)

echo
echo "Done."
echo "Total render time: ${TOTAL} seconds"