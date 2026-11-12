#!/bin/bash
# Run a command inside the DeepStream image with this folder at /work. Run on the GPU server from any directory:
#   scripts/docker.sh scripts/build.sh
# Mounts: /work (this folder), /plugins (EfficientNMSX_TRT plugin, read-only), /data (videos, ONNX, baseline, read-only).
set -e
cd "$(dirname "$0")/.."
: "${IMAGE:=ialy:latest}"
: "${PLUGIN_DIR:=/home/mq/disk2T/duy/ialy/vms_ialy/data/custom_path/nvdsinfer_custom_impl_Yolo_seg}"
: "${DATA_DIR:=/home/mq/disk2T/lele/ialy_test_engine}"
mkdir -p runs/tmp
exec docker run --rm --name "motion_ds_$$" --gpus all --log-driver none --user "$(id -u):$(id -g)" --entrypoint /bin/bash \
    -e HOME=/work/runs/tmp -e TMPDIR=/work/runs/tmp \
    -v "$PWD":/work -v "$PLUGIN_DIR":/plugins:ro -v "$DATA_DIR":/data:ro -w /work "$IMAGE" -c "$*"
