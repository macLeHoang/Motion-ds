#!/bin/bash
# fire.mp4 (2560x1440, 29.97 fps, clock at the top centre) through the pipeline, twice:
#   default      fire/smoke from full-frame slots only (every full-frame-interval frames)
#   mosaic_fire  fire/smoke also from the motion mosaics (mosaic-classes=0;4;5)
# then the timeline against full-frame inference on every frame ($BASELINE, made by infer_video.py).
# Inside the DeepStream image: scripts/docker.sh scripts/run_fire_test.sh
set -e
cd /work
V=/data/video_test/fire.mp4
BASELINE=${BASELINE:-/data/output/cmp/py768/fire_engine_dets.jsonl}
MASK='mask-polygons=[[[0.32,0],[0.68,0],[0.68,0.08],[0.32,0.08]]]'
COMMON=(--config configs/app.yaml --mux 2560x1440 --set "$MASK" --input "$V")
python3 app/motion_ds_app.py "${COMMON[@]}" --output runs/fire_default > runs/fire_default.log 2>&1
python3 app/motion_ds_app.py "${COMMON[@]}" --set "mosaic-classes=0;4;5" --output runs/fire_mosaic_fire \
    > runs/fire_mosaic_fire.log 2>&1
python3 analysis/fire_timeline.py --fps 29.97 full768="$BASELINE" default=runs/fire_default/fire_engine_dets.jsonl \
    mosaic_fire=runs/fire_mosaic_fire/fire_engine_dets.jsonl | tee runs/fire_timeline.md
