#!/bin/bash
# The library's hardware acceptance test (static / moving / mixed clips through nvdspreprocess + an identity nvinfer).
# Inside the DeepStream image: scripts/docker.sh scripts/acceptance.sh [SOURCES=8]
# Needs models/identity/identity.onnx, made where the `onnx` package is installed:
#   python native/motion_preprocess/tools/create_identity.py --output models/identity --onnx-only
set -e
cd /work
E=models/identity/identity.engine
[ -f $E ] || /usr/src/tensorrt/bin/trtexec --onnx=models/identity/identity.onnx --saveEngine=$E \
    --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --skipInference \
    > runs/identity_build.log 2>&1
python3 native/motion_preprocess/tools/runtime_smoke.py --build-dir build/motion \
    --config native/motion_preprocess/configs/preprocess_dgpu.ini --work-dir runs/acceptance \
    --identity-engine $E --sources "${1:-8}"
