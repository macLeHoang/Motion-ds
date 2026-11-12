#pragma once

#include "geometry.hpp"

#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <nvbufsurface.h>

namespace vms_motion {
void cuda_check(cudaError_t result, const char* operation);

struct RgbaView {
    const unsigned char* data = nullptr;
    std::size_t pitch = 0;
    int width = 0, height = 0;
    cudaTextureObject_t texture = 0;
};

// One scoped mapping per input-buffer use. No raw EGL mappings survive source removal.
class SurfaceView {
public:
    SurfaceView(NvBufSurface* surface, unsigned index);
    ~SurfaceView();
    SurfaceView(const SurfaceView&) = delete;
    SurfaceView& operator=(const SurfaceView&) = delete;

    const RgbaView& get() const {
        return view_;
    }

private:
    RgbaView view_;
    NvBufSurface* surface_ = nullptr;
    unsigned index_ = 0;
    void* resource_ = nullptr;
    bool mapped_egl_ = false;
};

class Gpu {
public:
    Gpu();
    ~Gpu();
    Gpu(const Gpu&) = delete;
    Gpu& operator=(const Gpu&) = delete;
    std::vector<unsigned char> motion_gray(const RgbaView&, int width, int height);
    void tensor(
        const RgbaView&,
        const Plan&,
        void* output,
        bool fp16,
        bool bgr,
        const std::array<float, 3>& scale,
        const std::array<float, 3>& offset
    );
    std::vector<unsigned char> rgba(const RgbaView&); // Preview only: full-frame host download.
    std::vector<unsigned char> canvas(const RgbaView&, const Plan&); // Preview only.
    void synchronize();

private:
    void reserve_small(std::size_t bytes);
    cudaStream_t stream_ = nullptr;
    unsigned char* small_ = nullptr;
    unsigned char* host_ = nullptr;
    std::size_t small_bytes_ = 0;
    Placement* tiles_ = nullptr;
    std::size_t tile_capacity_ = 0;
};
}
