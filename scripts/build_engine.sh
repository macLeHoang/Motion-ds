#!/bin/bash
# Build an FP16 TensorRT engine from a trt_nms ONNX, same recipe as vms_ialy scripts/build_engines.sh (EfficientNMSX_TRT
# plugin, cv6 output Conv pinned to FP16 on T4). Inside the DeepStream image:
#   scripts/docker.sh scripts/build_engine.sh ONNX ENGINE [SIZE=768] [MAX_BATCH=16]
set -e
ONNX=$1 ENGINE=$2 SIZE=${3:-768} MAXB=${4:-16}
OPT=$(( MAXB < 8 ? MAXB : 8 ))
mkdir -p "$(dirname "$ENGINE")"
/usr/src/tensorrt/bin/trtexec --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 \
    --minShapes=images:1x3x${SIZE}x${SIZE} --optShapes=images:${OPT}x3x${SIZE}x${SIZE} \
    --maxShapes=images:${MAXB}x3x${SIZE}x${SIZE} \
    --precisionConstraints=prefer --layerOutputTypes=/model/model.23/cv6/out/Conv:fp16 \
    --staticPlugins=/plugins/libnvdsinfer_custom_impl_Yolo_seg.so
ls -la "$ENGINE"
