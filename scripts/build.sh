#!/bin/bash
# Build the motion library with its tests and the NvDsInferYoloAttr parser into build/, then run the tests.
# Inside the DeepStream image: scripts/docker.sh scripts/build.sh
set -e
cd /work
cmake -S native/motion_preprocess -B build/motion -DVMS_WITH_DEEPSTREAM=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-75}" -Dnlohmann_json_DIR=/work/third_party/nlohmann_json
cmake --build build/motion -j"$(nproc)"
mkdir -p build/parser
cp third_party/nvdsinfer_yolo_attr/{Makefile,nvdsinfer_yolo_attr.cpp} build/parser/
make -C build/parser CUDA_VER="${CUDA_VER:-12.8}" NMSX_LIB=/plugins/libnvdsinfer_custom_impl_Yolo_seg.so
ctest --test-dir build/motion --output-on-failure
