#!/bin/bash
# Compare a DeepStream run with full-frame inference, per input. Inside the DeepStream image:
#   scripts/docker.sh scripts/analyse.sh RUN_DIR STEM [STEM ...]      (stems in --input order = source ids)
# Full-frame 768 baseline (infer_video.py) is read from $BASELINE (default /data/output/cmp/py768).
# Helmet-worn frame ranges per stem come from HELMET_<stem prefix> below (camera-1: 1000-3000, camera-4: 500-2000).
set -e
cd /work
RUN=$1; shift
BASE=${BASELINE:-/data/output/cmp/py768}
OUT=$RUN/analysis
mkdir -p $OUT
src=0
for V in "$@"; do
  case $V in
    camera-1-*) H="--helmet-frames 1000:3000" ;;
    camera-4-*) H="--helmet-frames 500:2000" ;;
    *) H="" ;;
  esac
  python3 analysis/motion_gate.py "$V" --full $BASE --mosaic $RUN --plans $RUN/plans --source $src $H \
    > $OUT/${V}_vs_full.md
  python3 analysis/compare_runs.py full768=$BASE/${V}_person_raw.npz deepstream=$RUN/${V}_person_raw.npz \
    > $OUT/${V}_compare.md
  python3 analysis/person_stats.py $RUN/${V}_person_raw.npz > /dev/null
  python3 analysis/helmet_stats.py $RUN/${V}_person_raw.npz > $OUT/${V}_helmet.md
  src=$((src + 1))
done
ls -la $OUT
