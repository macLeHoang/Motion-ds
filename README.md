# motion_ds_test — DeepStream motion-mosaic test

Runs the AIO model (persons + attributes + children, fire, smoke, vehicles, plates) on DeepStream 8.0 with motion-only
inputs: instead of shrinking the whole 1280×720 frame into 768×768 (scale 0.6), each camera's moving regions and tracked
people are cropped at native resolution, packed into one near-square mosaic and zoomed up to 2×. A full frame is still
inferred every N frames per camera for fire, smoke, vehicles and plates, and to find people who never moved.

```
nvurisrcbin ×cams → nvstreammux (1280×720) → nvvideoconvert (RGBA)
  → nvdspreprocess   motion library: per camera a mosaic slot (motion boxes ∪ person regions),
                     + a full-frame slot every N frames (cameras staggered); one tensor, one plan per slot
  → nvinfer          AIO 768 engine on that tensor (input-tensor-from-meta, raw outputs)
  → decode probe     C++: NvDsInferYoloAttr per slot → boxes mapped back to the frame → mosaic slots keep persons,
                     full-frame slots keep fire/smoke/vehicles/plates (+ persons not covered by the mosaic)
                     → objects with the production payload → person boxes fed back to the regions
  → nvtracker        production NvDCF config
  → yolo_attr_probe  attribute labels, child boxes, plate corners (+ the test dump)
  → nvdsosd → one mp4 per camera
```

## Layout

| Path | What |
|---|---|
| `native/motion_preprocess/` | The `nvdspreprocess` custom library (from `vmsx-dls-vms-2026.10.09`) with the changes below |
| `dls/core/` | Python original of the motion detector, only for the library's `python_reference` test |
| `third_party/nvdsinfer_yolo_attr/` | `NvDsInferYoloAttr` parser + `yolo_attr_probe.py`, unchanged from Ultralytics-Attri |
| `third_party/nlohmann_json/` | JSON header the library needs |
| `app/motion_ds_app.py` | The DeepStream pipeline (pyds) |
| `configs/` | `preprocess.ini` (motion, N, K, regions, classes), `infer_mosaic.txt`, `app.yaml` (mux, paths), `tracker/`, `meta.json`, `labels.txt` |
| `analysis/` | Comparison with full-frame inference: `motion_gate.py`, `compare_runs.py`, `person_stats.py`, `helmet_stats.py` |
| `scripts/` | `docker.sh` (run anything in the DeepStream image), `build.sh`, `build_engine.sh`, `acceptance.sh`, `analyse.sh` |
| `models/`, `runs/` | Engines and outputs (git-ignored) |

The first commit holds every vendored file unchanged; `git log -p` shows each change.

## Run (on the GPU server, from this folder)

```bash
scripts/docker.sh scripts/build.sh                     # library + tests + parser into build/, runs ctest
scripts/docker.sh scripts/build_engine.sh /data/model/AIO_t13_11_11_768-trt.onnx models/AIO_t13_11_11_768_b16.engine 768 16
scripts/docker.sh scripts/acceptance.sh 8              # library acceptance test (needs models/identity/identity.onnx)

V=/data/video_test
scripts/docker.sh python3 app/motion_ds_app.py --config configs/app.yaml --output runs/my_run \
    --input $V/camera-1-20261111-221001-20261111-221701.mp4 --input $V/camera-2-20261111-221100-20261111-221500.mp4
scripts/docker.sh scripts/analyse.sh runs/my_run camera-1-20261111-221001-20261111-221701 camera-2-20261111-221100-20261111-221500
```

`scripts/docker.sh` mounts this folder at `/work`, the EfficientNMSX plugin at `/plugins` and
`/home/mq/disk2T/lele/ialy_test_engine` (videos, ONNX, full-frame baseline) at `/data`; override with `IMAGE`,
`PLUGIN_DIR`, `DATA_DIR`. Containers are named `motion_ds_<pid>` (`docker stop` them to abort).

A run folder holds `<stem>_ds.mp4` (OSD), `<stem>_engine_dets.jsonl` (objects after the tracker, with origin
`mosaic`/`full` and track id), `<stem>_person_raw.npz` (persons with decoded attributes and children),
`plans/events_1.jsonl` (every frame's plan: slots, crops, regions, timings), `plans/*.png` (library snapshots of the
mosaic input) and `summary.json`; `analyse.sh` adds `analysis/`.

## Configuration

| What | Where | Key | Test value |
|---|---|---|---|
| Mux size (frame coordinates of crops and boxes) | `configs/app.yaml` | `muxer.width`, `muxer.height` | 1280×720 (native) |
| Full frame every N frames per camera; 0 = never | `configs/preprocess.ini` | `full-frame-interval` | 25 (1 s) |
| Cameras take turns for the full frame | `preprocess.ini` | `full-frame-stagger` | 1 |
| K: inferred frames without a person before a region is removed | `preprocess.ini` | `region-max-misses` | 25 |
| Region growth / match / cap | `preprocess.ini` | `region-padding`, `region-match-iou`, `max-regions` | 0.2, 0.3, 16 |
| Person score to refresh a region (mosaic) / seed one (full frame) | `preprocess.ini` | `region-score`, `seed-score` | 0.5, 0.5 |
| Classes from mosaic slots / full-frame slots | `preprocess.ini` | `mosaic-classes`, `full-frame-classes` | `0` / `1;2;3;4;5;6` |
| Detection threshold, cross-crop dedupe | `preprocess.ini` | `det-score`, `dedupe-iou` | 0.25, 0.5 |
| Max zoom of the mosaic | `preprocess.ini` | `max-zoom` | 2 |
| Motion masks (normalised polygons) | `preprocess.ini` | `mask-polygons`, `source-masks` | camera clock |
| Tensor slots per batch (≥ cameras + cameras due for a full frame) | `preprocess.ini` | `network-input-shape` | 16 |
| Engine / nvinfer batch | `configs/infer_mosaic.txt` | `model-engine-file`, `batch-size` | 768, 16 |
| OSD video on/off | `app.yaml` or `--osd on/off` | `osd` | on |

`preprocess.ini` documents every key; the motion keys are the library's originals (see its README).

## How it works

- **Slots.** For each frame the library makes at most two tensor slots: a mosaic (if there is motion or a region) and,
  when `(frame + camera % N) % N == 0`, a full frame. The full frame is a plan with one tile covering the frame, so it
  goes through the same CUDA kernel and gives exactly the Ultralytics letterbox (gray 114, bilinear) — unlike
  nvinfer's own scaling (black padding, nearest). nvinfer returns slot j's outputs on the preprocess `roi_vector[j]`.
- **Regions.** A confident person from a mosaic refreshes the region it overlaps (or opens one); a region with no
  person inside for K inferred mosaics is removed; a confident full-frame person opens or refreshes a region but never
  counts a miss. Regions become crops (with padding) next to the motion boxes, so a person who stops moving stays in the
  mosaic, and is cropped whole rather than only the moving arm.
- **Decode probe** (`native/motion_preprocess/src/decoder.cpp`): `NvDsInferYoloAttr` on each slot's raw outputs; boxes
  mapped back through the tile holding their centre (not clipped to the crop, so the box-relative payload stays exact);
  objects carry the same 160-float payload in `mask_params` as production `NvDsInferYoloAttr` objects, origin in
  `misc_obj_info[0]` (1 mosaic, 2 full frame). It sets `bInferDone` on inferred frames (nvtracker ignores objects
  otherwise) and queues region feedback that the next batch applies (it must not wait on the preprocessing lock).
- **Tracker.** Production NvDCF config. Between full frames the tracker draws predicted boxes for full-frame classes
  (e.g. plates); those carry no payload and are skipped by `yolo_attr_probe` and the dump.

## Changes to the vendored library

| Commit | Change |
|---|---|
| `ae6e648` | Calibration lasts until the contrast history is full (fixes false motion on static scenes for ~2 s after every start/reset; the library's own acceptance test failed on it). Same fix in `dls/core/motion.py`. |
| `f10ecb9` | Person regions, staggered full-frame slots, plan metadata v2, decode probe, `vms_motion_install_probes`; DeepStream 8.0 build fixes; tests |
| `6736662` | Region feedback through a queue (fixes a deadlock), `bInferDone`, plan log flushed per batch |

`git diff d74eec2 -- native dls > motion_preprocess.patch` gives the patch against the original.

## Results (2026-11-12, Tesla T4, DeepStream 8.0, AIO_t13_11_11 768 FP16)

camera-1, camera-2 and camera-4 (1280×720, 25 fps; 10491 + 7242 + 4449 frames) as three cameras in one pipeline,
N = 25, K = 25, mux 1280×720. Reference: full-frame 768 inference on every frame with the Ultralytics letterbox
(`infer_video.py`, `ialy_test_engine/output/cmp/py768`). Run folders on the server: `runs/main_3cam` (tracker, OSD),
`runs/main_3cam_notracker` (detections before the tracker), `runs/bench_*` (throughput).

| | camera-1 | camera-2 | camera-4 |
|---|---|---|---|
| Inferences per frame (mosaic + full frame) | 22 % | 47 % | 74 % |
| Person height in the model input, median (full frame → mosaic) | 126 → 416 px | 118 → 378 px | 108 → 348 px |
| Confident full-frame persons found, before the tracker | 98.1 % | 95.1 % | 98.7 % |
| ... after the production NvDCF tracker | 95.1 % | 89.5 % | 95.7 % |
| ... earlier motion-only mosaic (no regions, no full frame) | 96.4 % | 79.8 % | 92.1 % |
| Confident persons missed in idle frames (now / motion-only) | 0 / 4 | 0 / 6 | 0 / 95 |
| Helmet worn: `has_helm` > 0.5, full frame → this pipeline | 11.7 → 56.2 % | – | 21.3 → 34.8 % |
| Bareheaded: `has_helm` > 0.5, full frame → this pipeline | 0 → 3.3 % | 0 → 0.3 % | 0 → 0.1 % |
| COCO / fire false positives from mosaics | 0 | 0 | 0 |

- Regions keep still people in the mosaic: no confident person is lost in an idle frame any more, and crops cover the
  whole person (camera-2 detections 80 → 95 %). The remaining detection misses are borderline (full-frame score
  median 0.55–0.57).
- The production tracker drops detections below `minDetectorConfidence: 0.4` and the first `probationAge: 5` frames of
  every track, about 12 % of the mosaic persons here; that, not the motion path, is the gap after the tracker.
- Mosaic slots keep persons only, so the zoomed-crop false positives (fire on hair, COCO classes) are gone. The static
  `license_plate` on the blue wall sign still comes from every full frame (one per 25 frames instead of every frame).
- Throughput, 3 cameras with the Python dump and `yolo_attr_probe` probes: 204 fps with OSD video, 212 fps without
  (≈ 70 fps per camera, 2.8× real time); the library adds 2–4 ms per active frame (motion + tensor). Keep
  `enable-last-sample=false` on a fakesink: a held batch pins one of the `tensor-buf-pool-size` tensor buffers
  (154 fps with it on).
- Library acceptance test (`scripts/acceptance.sh 8`) passes fully: static clip 0 active frames (21 before the
  warm-up fix), moving clip exactly its 140 moving frames, 8-camera mix 560.

Known limits: nvdsosd does not print the attribute labels that `yolo_attr_probe.attach` adds as classifier metadata
(they are in the metadata and the dumps); nvinfer's full-frame ROI shows as a thin green frame border on the OSD;
regions follow detections one to three frames late (asynchronous feedback, covered by `region-padding`).
