FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake g++ pkg-config libopencv-dev libgstreamer1.0-dev nlohmann-json3-dev \
    python3-opencv python3-numpy python3-onnx ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY native/motion_preprocess /work/native/motion_preprocess
COPY dls/core /work/dls/core
ARG CUDA_ARCHITECTURES="75;80;86;89"
RUN cmake -S native/motion_preprocess -B /work/build \
    -DVMS_WITH_DEEPSTREAM=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
    && cmake --build /work/build -j2
ENTRYPOINT ["/work/build/vms_motion_preview"]
