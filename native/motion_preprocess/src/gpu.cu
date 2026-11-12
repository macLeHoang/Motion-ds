#include "vms_motion/gpu.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <cuda.h>
#include <cudaEGL.h>
#include <cuda_fp16.h>

namespace vms_motion {
void cuda_check(cudaError_t r, const char* op) {
    if (r != cudaSuccess)
        throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(r));
}

static void driver_check(CUresult r, const char* op) {
    if (r != CUDA_SUCCESS) {
        const char* error = nullptr;
        cuGetErrorString(r, &error);
        throw std::runtime_error(std::string(op) + ": " + (error ? error : "CUDA driver error"));
    }
}

SurfaceView::SurfaceView(NvBufSurface* s, unsigned index)
    : surface_(s),
      index_(index) {
    if (!s || index >= s->numFilled)
        throw std::invalid_argument("surface index out of range");
    auto& p = s->surfaceList[index];
    if (p.colorFormat != NVBUF_COLOR_FORMAT_RGBA)
        throw std::invalid_argument("motion preprocessing requires NVMM RGBA input");
    view_.width = p.width;
    view_.height = p.height;
    if (s->memType == NVBUF_MEM_SURFACE_ARRAY) {
        if (NvBufSurfaceMapEglImage(s, index) != 0)
            throw std::runtime_error("NvBufSurfaceMapEglImage failed");
        mapped_egl_ = true;
        CUgraphicsResource resource = nullptr;
        try {
            driver_check(
                cuGraphicsEGLRegisterImage(
                    &resource, p.mappedAddr.eglImage, CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY
                ),
                "register EGL image"
            );
            resource_ = resource;
            CUeglFrame f{};
            driver_check(cuGraphicsResourceGetMappedEglFrame(&f, resource, 0, 0), "map EGL frame");
            if (f.frameType == CU_EGL_FRAME_TYPE_PITCH) {
                view_.data = static_cast<unsigned char*>(f.frame.pPitch[0]);
                view_.pitch = f.pitch;
            } else {
                cudaResourceDesc rd{};
                rd.resType = cudaResourceTypeArray;
                rd.res.array.array = reinterpret_cast<cudaArray_t>(f.frame.pArray[0]);
                cudaTextureDesc td{};
                td.readMode = cudaReadModeElementType;
                td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
                td.filterMode = cudaFilterModePoint;
                td.normalizedCoords = 0;
                cuda_check(
                    cudaCreateTextureObject(&view_.texture, &rd, &td, nullptr), "create EGL texture"
                );
            }
        } catch (...) {
            if (resource_)
                cuGraphicsUnregisterResource(static_cast<CUgraphicsResource>(resource_));
            NvBufSurfaceUnMapEglImage(s, index);
            mapped_egl_ = false;
            throw;
        }
    } else if (
        s->memType == NVBUF_MEM_CUDA_DEVICE || s->memType == NVBUF_MEM_CUDA_UNIFIED ||
        s->memType == NVBUF_MEM_DEFAULT
    ) {
        view_.data = static_cast<unsigned char*>(p.dataPtr);
        view_.pitch = p.pitch;
        if (!view_.data || view_.pitch < std::size_t(view_.width) * 4)
            throw std::invalid_argument("invalid CUDA RGBA surface");
    } else
        throw std::invalid_argument("unsupported surface memory type");
}

SurfaceView::~SurfaceView() {
    if (view_.texture)
        cudaDestroyTextureObject(view_.texture);
    if (resource_)
        cuGraphicsUnregisterResource(static_cast<CUgraphicsResource>(resource_));
    if (mapped_egl_)
        NvBufSurfaceUnMapEglImage(surface_, index_);
}

__device__ uchar4 pixel(RgbaView v, int x, int y) {
    x = max(0, min(x, v.width - 1));
    y = max(0, min(y, v.height - 1));
    if (v.texture)
        return tex2D<uchar4>(v.texture, x + .5f, y + .5f);
    return *reinterpret_cast<const uchar4*>(v.data + y * v.pitch + x * 4);
}

__global__ void gray_kernel(RgbaView v, unsigned char* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;
    auto p = pixel(
        v,
        int(static_cast<long long>(x) * v.width / w),
        int(static_cast<long long>(y) * v.height / h)
    );
    // OpenCV's 14-bit fixed-point RGB -> gray conversion, including rounding.
    out[y * w + x] =
        static_cast<unsigned char>((p.x * 4899 + p.y * 9617 + p.z * 1868 + 8192) >> 14);
}

struct CanvasGeometry {
    int mw, mh, rw, rh, px, py, cw, ch;
};

__device__ uchar4 mosaic_pixel(RgbaView v, const Placement* tiles, int count, int x, int y) {
    for (int i = 0; i < count; ++i) {
        auto t = tiles[i];
        if (x >= t.x && y >= t.y && x < t.x + t.w && y < t.y + t.h)
            return pixel(v, t.source.x1 + x - t.x, t.source.y1 + y - t.y);
    }
    return make_uchar4(114, 114, 114, 255);
}

__device__ uchar4
canvas_pixel(RgbaView v, const Placement* tiles, int count, CanvasGeometry g, int x, int y) {
    if (x < g.px || y < g.py || x >= g.px + g.rw || y >= g.py + g.rh)
        return make_uchar4(114, 114, 114, 255);
    float fx = (x - g.px + .5f) * g.mw / g.rw - .5f, fy = (y - g.py + .5f) * g.mh / g.rh - .5f;
    int ix = int(floorf(fx)), iy = int(floorf(fy));
    float ax = fx - ix, ay = fy - iy;
    uchar4 a = mosaic_pixel(v, tiles, count, max(0, min(ix, g.mw - 1)), max(0, min(iy, g.mh - 1)));
    uchar4 b =
        mosaic_pixel(v, tiles, count, max(0, min(ix + 1, g.mw - 1)), max(0, min(iy, g.mh - 1)));
    uchar4 c =
        mosaic_pixel(v, tiles, count, max(0, min(ix, g.mw - 1)), max(0, min(iy + 1, g.mh - 1)));
    uchar4 d =
        mosaic_pixel(v, tiles, count, max(0, min(ix + 1, g.mw - 1)), max(0, min(iy + 1, g.mh - 1)));
    uchar4 out;
    auto* dst = reinterpret_cast<unsigned char*>(&out);
    auto* pa = reinterpret_cast<unsigned char*>(&a);
    auto* pb = reinterpret_cast<unsigned char*>(&b);
    auto* pc = reinterpret_cast<unsigned char*>(&c);
    auto* pd = reinterpret_cast<unsigned char*>(&d);
    for (int channel = 0; channel < 3; ++channel)
        dst[channel] = static_cast<unsigned char>(__float2int_rn(
            (pa[channel] * (1 - ax) + pb[channel] * ax) * (1 - ay) +
            (pc[channel] * (1 - ax) + pd[channel] * ax) * ay
        ));
    out.w = 255;
    return out;
}

template <class T> __device__ T cast_value(float x) {
    return T(x);
}

template <> __device__ __half cast_value<__half>(float x) {
    return __float2half_rn(x);
}

template <class T>
__global__ void tensor_kernel(
    RgbaView v,
    const Placement* tiles,
    int count,
    CanvasGeometry g,
    T* out,
    bool bgr,
    float3 scale,
    float3 offset
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= g.cw || y >= g.ch)
        return;
    auto p = canvas_pixel(v, tiles, count, g, x, y);
    int n = g.cw * g.ch, at = y * g.cw + x;
    out[at] = cast_value<T>((bgr ? p.z : p.x) * scale.x + offset.x);
    out[n + at] = cast_value<T>(p.y * scale.y + offset.y);
    out[2 * n + at] = cast_value<T>((bgr ? p.x : p.z) * scale.z + offset.z);
}

__global__ void rgba_kernel(RgbaView v, uchar4* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < v.width && y < v.height)
        out[y * v.width + x] = pixel(v, x, y);
}

__global__ void
canvas_kernel(RgbaView v, const Placement* tiles, int count, CanvasGeometry g, uchar4* out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < g.cw && y < g.ch)
        out[y * g.cw + x] = canvas_pixel(v, tiles, count, g, x, y);
}

static CanvasGeometry geometry(const Plan& p) {
    return {
        p.mosaic_w, p.mosaic_h, p.resized_w, p.resized_h, p.pad_x, p.pad_y, p.canvas_w, p.canvas_h
    };
}

Gpu::Gpu() {
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream");
}

Gpu::~Gpu() {
    cudaStreamSynchronize(stream_);
    if (small_)
        cudaFree(small_);
    if (host_)
        cudaFreeHost(host_);
    if (tiles_)
        cudaFree(tiles_);
    cudaStreamDestroy(stream_);
}

void Gpu::synchronize() {
    cuda_check(cudaStreamSynchronize(stream_), "synchronize");
}

void Gpu::reserve_small(std::size_t bytes) {
    if (bytes <= small_bytes_)
        return;
    synchronize();
    if (small_)
        cudaFree(small_);
    if (host_)
        cudaFreeHost(host_);
    small_ = nullptr;
    host_ = nullptr;
    small_bytes_ = 0;
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&small_), bytes), "allocate device staging");
    try {
        cuda_check(
            cudaMallocHost(reinterpret_cast<void**>(&host_), bytes), "allocate host staging"
        );
    } catch (...) {
        cudaFree(small_);
        small_ = nullptr;
        throw;
    }
    small_bytes_ = bytes;
}

std::vector<unsigned char> Gpu::motion_gray(const RgbaView& v, int w, int h) {
    if (w <= 0 || h <= 0)
        throw std::invalid_argument("invalid motion size");
    std::size_t bytes = std::size_t(w) * h;
    reserve_small(bytes);
    gray_kernel<<<dim3((w + 15) / 16, (h + 15) / 16), dim3(16, 16), 0, stream_>>>(v, small_, w, h);
    cuda_check(cudaGetLastError(), "motion kernel");
    cuda_check(
        cudaMemcpyAsync(host_, small_, bytes, cudaMemcpyDeviceToHost, stream_),
        "download small gray"
    );
    synchronize();
    return {host_, host_ + bytes};
}

void Gpu::tensor(
    const RgbaView& v,
    const Plan& p,
    void* out,
    bool fp16,
    bool bgr,
    const std::array<float, 3>& s,
    const std::array<float, 3>& o
) {
    if (!out || p.tiles.empty())
        throw std::invalid_argument("cannot tensorize empty canvas");
    if (p.tiles.size() > tile_capacity_) {
        synchronize();
        cudaFree(tiles_);
        tiles_ = nullptr;
        tile_capacity_ = 0;
        cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&tiles_), p.tiles.size() * sizeof(Placement)),
            "allocate tiles"
        );
        tile_capacity_ = p.tiles.size();
    }
    cuda_check(
        cudaMemcpyAsync(
            tiles_,
            p.tiles.data(),
            p.tiles.size() * sizeof(Placement),
            cudaMemcpyHostToDevice,
            stream_
        ),
        "upload tiles"
    );
    auto grid = dim3((p.canvas_w + 15) / 16, (p.canvas_h + 15) / 16);
    auto block = dim3(16, 16);
    auto scale = make_float3(s[0], s[1], s[2]), offset = make_float3(o[0], o[1], o[2]);
    if (fp16)
        tensor_kernel<<<grid, block, 0, stream_>>>(
            v,
            tiles_,
            int(p.tiles.size()),
            geometry(p),
            static_cast<__half*>(out),
            bgr,
            scale,
            offset
        );
    else
        tensor_kernel<<<grid, block, 0, stream_>>>(
            v,
            tiles_,
            int(p.tiles.size()),
            geometry(p),
            static_cast<float*>(out),
            bgr,
            scale,
            offset
        );
    cuda_check(cudaGetLastError(), "tensor kernel");
    // Complete before pool ownership is handed to nvdspreprocess and before EGL unmapping.
    synchronize();
}

std::vector<unsigned char> Gpu::rgba(const RgbaView& v) {
    std::size_t bytes = std::size_t(v.width) * v.height * 4;
    reserve_small(bytes);
    rgba_kernel<<<dim3((v.width + 15) / 16, (v.height + 15) / 16), dim3(16, 16), 0, stream_>>>(
        v, reinterpret_cast<uchar4*>(small_)
    );
    cuda_check(cudaGetLastError(), "preview source kernel");
    cuda_check(
        cudaMemcpyAsync(host_, small_, bytes, cudaMemcpyDeviceToHost, stream_),
        "download preview source"
    );
    synchronize();
    return {host_, host_ + bytes};
}

std::vector<unsigned char> Gpu::canvas(const RgbaView& v, const Plan& p) {
    if (p.tiles.empty() || p.canvas_w <= 0 || p.canvas_h <= 0)
        throw std::invalid_argument("cannot preview an empty placement plan");
    // tensor() normally uploads placements first, but preview is also independently usable.
    if (p.tiles.size() > tile_capacity_) {
        synchronize();
        cudaFree(tiles_);
        tiles_ = nullptr;
        tile_capacity_ = 0;
        cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&tiles_), p.tiles.size() * sizeof(Placement)),
            "allocate preview tiles"
        );
        tile_capacity_ = p.tiles.size();
    }
    cuda_check(
        cudaMemcpyAsync(
            tiles_,
            p.tiles.data(),
            p.tiles.size() * sizeof(Placement),
            cudaMemcpyHostToDevice,
            stream_
        ),
        "upload preview tiles"
    );
    std::size_t bytes = std::size_t(p.canvas_w) * p.canvas_h * 4;
    reserve_small(bytes);
    canvas_kernel<<<
        dim3((p.canvas_w + 15) / 16, (p.canvas_h + 15) / 16),
        dim3(16, 16),
        0,
        stream_>>>(v, tiles_, int(p.tiles.size()), geometry(p), reinterpret_cast<uchar4*>(small_));
    cuda_check(cudaGetLastError(), "preview canvas kernel");
    cuda_check(
        cudaMemcpyAsync(host_, small_, bytes, cudaMemcpyDeviceToHost, stream_),
        "download preview canvas"
    );
    synchronize();
    return {host_, host_ + bytes};
}
}
