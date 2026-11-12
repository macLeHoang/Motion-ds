# Motion-mosaic preprocessing

A reusable C++/CUDA library for DeepStream 7.1 `nvdspreprocess`. It ports the motion detector and near-square crop layout from `dls/core`. No trained model is needed to preview its output.

```
Decoded video → RGBA NVMM → nvstreammux → nvdspreprocess → preview/fakesink
                                            │
                                            └─ active-camera tensors + placement metadata
```

The original video buffer continues downstream. Only frames with usable motion regions create tensor samples. Calibration and idle frames create no tensor, and there are no periodic full-frame scans. Objects that remain still may eventually be absorbed into the background; detection feedback and recent-object retention are outside this version.

## Build

CPU geometry tests need CMake and a C++17 compiler. Motion tests additionally need OpenCV development files. Python reference comparisons need NumPy and OpenCV Python bindings.

```bash
cmake -S native/motion_preprocess -B /tmp/vms-motion-build -DCMAKE_BUILD_TYPE=Debug
cmake --build /tmp/vms-motion-build -j2
ctest --test-dir /tmp/vms-motion-build --output-on-failure
```

Without OpenCV development files, only native geometry and its Python reference comparisons run. The reference runner explicitly reports skipped native motion comparisons.

For the complete library, build inside the repository's DeepStream 7.1 development container, or on a Jetson with DeepStream 7.1 and its compatible JetPack installed:

```bash
sudo apt-get update
sudo apt-get install -y cmake g++ pkg-config libopencv-dev libgstreamer1.0-dev nlohmann-json3-dev
cmake -S native/motion_preprocess -B native/motion_preprocess/build \
  -DVMS_WITH_DEEPSTREAM=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build native/motion_preprocess/build -j2
ctest --test-dir native/motion_preprocess/build --output-on-failure
```

Architecture 87 is for Jetson Orin. Set the architecture to your GPU for desktop/server builds; the supplied Dockerfile builds architectures 75, 80, 86, and 89. Override `DEEPSTREAM_ROOT` if your SDK is installed elsewhere.

An independent desktop/server Docker build is available; run these commands from the repository root:

```bash
docker build -f native/motion_preprocess/Dockerfile -t vms-motion-preprocess .
docker run --rm --gpus all \
  -v "$PWD/data:/data:ro" -v "$PWD/motion-preview:/preview" \
  vms-motion-preprocess \
  --config /work/native/motion_preprocess/configs/preprocess_dgpu.ini \
  --input /data/your-clip.mp4 --output /preview
```

## Preview

Use the platform-specific config. Repeat `--input` for multiple file paths or RTSP URIs. The sample sets the mux batch size to the number of sources, rewrites the source group, and finds the loaded shared library automatically.

```bash
native/motion_preprocess/build/vms_motion_preview \
  --config native/motion_preprocess/configs/preprocess_jetson.ini \
  --input data/clip.mp4 --output motion-preview --verify-pixels
```

`--width` and `--height` configure the mux output (default 1280×720). These dimensions define the source coordinate space in placement metadata. Files are processed without clock synchronization. Queues are non-leaky so all delivered frames update motion. The sample requires the legacy DeepStream 7.1 mux; unset `USE_NEW_NVSTREAMMUX`.

The default preview rate is one snapshot per stream second per camera. Each snapshot contains `.source.png` with crop outlines, `.mosaic.png`, and placement `.json`. Idle/calibrating previews show a status card, never an old mosaic. `events_<component>.jsonl` records frame status and timings, including small-image download in motion timing. PNG/download/write costs are additional preview overhead, excluded from those timings.

Use `--output ""` to disable preview downloads and disk writes. `--verify-pixels` adds a full-frame checksum before and after preprocessing; use it for acceptance testing, not performance measurement. The runner prints JSON counters on completion. `--max-frames N` limits the total number of frames delivered across sources.

## Configuration and integration contract

The two INI templates use RGB, FP32, NCHW, `pixel / 255`, 640×640, and capacity eight. Jetson uses surface-array/VIC staging; desktop/server GPUs use CUDA-device/GPU staging. Original surfaces must be RGBA NVMM. Color conversion from NV12 belongs in `nvvideoconvert` before preprocessing.

Standard `[property]` settings select tensor shape, color order (RGB/BGR), type (FP32/FP16), input name, capacity, component ID and target inference IDs. Tensor capacity must cover the mux batch size. `[group-0]` must contain one full-frame unit per frame: `process-on-frame=1`, `process-on-roi=0`, and the configured source IDs.

Supported `[user-configs]` keys:

| Setting | Default | Meaning |
|---|---|---|
| `gpu-id` | `0` | Must agree with the preprocessing property and incoming surfaces |
| `motion-height`, `motion-threshold`, `contour-area` | `100`, `30`, `10` | Low-resolution detector settings |
| `frame-alpha`, `lightning-threshold` | `0.01`, `0.8` | Background absorption and scene-change threshold |
| `contrast-history`, `improve-contrast`, `stale-frames` | `50`, `1`, `0` | Contrast history and optional stale relearning |
| `crop-padding`, `gap`, `max-zoom`, `merge-growth` | `0.15`, `4`, `2`, `1.3` | Existing near-square packing behavior; merge growth `0` permits unrestricted unions |
| `channel-scale`, `channel-offset` | three copies of `1/255`, three zeros | Three semicolon-separated values in output channel order; output is `pixel * scale + offset` |
| `mask-polygons` | `[]` | JSON array of ignored polygons, each point `[x,y]` normalized to `[0,1]` |
| `source-masks` | `{}` | JSON object keyed by source ID; replaces global masks for those sources |
| `preview-dir`, `preview-fps` | empty, `1` | Optional debug output directory and snapshot rate |

Packing remains at native crop scale in a logical near-square mosaic. CUDA samples that layout directly into the final letterboxed image, avoiding a potentially huge intermediate allocation. The final effective horizontal/vertical resize factors account for integer rounding. Interpolation at crop borders also samples the neutral gaps, as resizing a materialized mosaic would.

Exported SDK functions are `initLib`, `deInitLib`, and `CustomTensorPreparation`. Active samples are compacted into contiguous slots; tensor shape `[N,3,H,W]`, byte count, and ROI references describe only those samples. Several frames from one source can be handled in the same batch. An entirely skipped batch returns `NVDSPREPROCESS_TENSOR_NOT_READY` without acquiring a tensor buffer.

**Install `vms_motion_strip_empty_tensors` on the preprocessing src pad before a future `nvinfer`.** Stock `nvdspreprocess` can attach metadata with `tensor_meta == nullptr` even when a tensor is not ready. This probe removes that metadata while forwarding the video buffer. Pass `GUINT_TO_POINTER(component_id)` as probe data to make missing placement metadata a pipeline error. The preview runner does this automatically.

For a future detector, enable `input-tensor-from-meta=1`, use a compatible explicit-batch engine/profile, and set its ID in `target-unique-ids`. No-frame/no-tensor behavior is verified using the test-only identity engine below. The standard bounding-box mapping is not a mosaic decoder: later integration must decode tensor outputs and map detections before tracking.

Placement metadata type is `VMS.MOTION_MOSAIC.PLACEMENTS.V1`, attached to each `NvDsFrameMeta` with deep-copy/release callbacks. `FramePlan` records version, component/source IDs, frame number, PTS, source dimensions, status, tensor slot (`-1` when skipped), timings, and the complete `Plan`. `frame_plan()` retrieves it. `map_box()` and `map_point()` accept final tensor/canvas coordinates before SDK ROI transformations; boxes are assigned by center and clipped to their source crop. Consumers own cross-crop deduplication.

Backgrounds reset automatically on dimension or timestamp/frame-number regression. Call thread-safe `vms_motion_reset_source(component_id, source_id)` on removal, reconnect, or stream reset. The sample connects mux EOS/removal/reset events. Treat IDs as camera lifecycle IDs rather than permanent background keys.

## Hardware acceptance

GPU CTest verifies pitched source memory, grayscale, canvas interpolation, RGB/BGR, FP32/FP16, normalization, unmodified source pixels, and metadata ownership. Jetson surface-array/EGL behavior is additionally exercised by the platform preview pipeline.

Install Python `onnx`, NumPy, and `ffmpeg` for runtime fixtures. The Dockerfile includes them. A tiny test-only identity network is generated locally; it is not an application detector. TensorRT engines must be built on the target GPU/runtime.

```bash
python3 native/motion_preprocess/tools/create_identity.py --output /tmp/vms-motion-identity
python3 native/motion_preprocess/tools/runtime_smoke.py \
  --build-dir native/motion_preprocess/build \
  --config native/motion_preprocess/configs/preprocess_jetson.ini \
  --work-dir /tmp/vms-motion-acceptance \
  --identity-engine /tmp/vms-motion-identity/identity.engine --sources 8
```

Use the desktop/server config on that platform. The identity fixture assumes the default `images` FP32 input and 640×640 dimensions. The test runs static, moving, and mixed-source clips; checks exact frame preservation and active sample counts; and confirms inference outputs are absent for skipped cameras. It fails if static footage creates active samples. Logs and previews remain in the work directory. `--sources` defaults to two and can exercise up to eight alternating static/moving cameras. `create_identity.py --onnx-only` validates the model fixture without TensorRT.

Native motion tests exercise independent sources, source resets, regression detection, and repeated state teardown. Before deployment, also run an eight-camera RTSP soak and repeated reconnect/start/stop cycles on each target; inspect GPU/pinned-memory growth and preview event timings. The current workspace lacks OpenCV development files, DeepStream, and NVCC, so only geometry and Python reference geometry checks have been executable here. Hardware acceptance is not claimed until those target checks pass.

SDK references: [custom preprocessing](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvdspreprocess.html), [custom-library API](https://docs.nvidia.com/metropolis/deepstream/7.1/sdk-api/group__gstreamer__nvdspreprocess__api.html), and [tensor-input inference](https://docs.nvidia.com/metropolis/deepstream/7.1/text/DS_plugin_gst-nvinfer.html).
