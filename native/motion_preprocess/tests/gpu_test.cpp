#include "vms_motion/gpu.hpp"

#include "vms_motion/metadata.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <cuda_fp16.h>
#include <gstnvdsmeta.h>
#include <nvdspreprocess_interface.h>
#include <nvdspreprocess_meta.h>
#include <opencv2/imgproc.hpp>

using namespace vms_motion;
extern "C" CustomCtx* initLib(CustomInitParams);
extern "C" void deInitLib(CustomCtx*);

static void check(bool b, const char* m) {
    if (!b)
        throw std::runtime_error(m);
}

int main() {
    unsigned char* device = nullptr;
    void* tensor = nullptr;
    try {
        gst_init(nullptr, nullptr);
        int gpu = 0;
        cuda_check(cudaGetDevice(&gpu), "get GPU");
        const int w = 80, h = 64, cw = 64, ch = 64;
        std::size_t pitch = 0;
        std::vector<unsigned char> original(w * h * 4);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                auto at = (y * w + x) * 4;
                original[at] = x * 3;
                original[at + 1] = y * 2;
                original[at + 2] = x + y;
                original[at + 3] = 255;
            }
        cuda_check(
            cudaMallocPitch(reinterpret_cast<void**>(&device), &pitch, w * 4, h),
            "allocate pitched source"
        );
        cuda_check(
            cudaMemcpy2D(device, pitch, original.data(), w * 4, w * 4, h, cudaMemcpyHostToDevice),
            "upload test image"
        );
        NvBufSurfaceParams params{};
        params.width = w;
        params.height = h;
        params.pitch = pitch;
        params.dataPtr = device;
        params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
        NvBufSurface surface{};
        surface.gpuId = gpu;
        surface.batchSize = 1;
        surface.numFilled = 1;
        surface.memType = NVBUF_MEM_CUDA_DEVICE;
        surface.surfaceList = &params;
        SurfaceView mapped(&surface, 0);
        Gpu kernels;
        auto view = mapped.get();
        auto gray = kernels.motion_gray(view, 20, 16);
        cv::Mat image(h, w, CV_8UC4, original.data()), gray_full, gray_small;
        cv::cvtColor(image, gray_full, cv::COLOR_RGBA2GRAY);
        cv::resize(gray_full, gray_small, {20, 16}, 0, 0, cv::INTER_NEAREST);
        check(
            cv::norm(gray_small, cv::Mat(16, 20, CV_8UC1, gray.data()), cv::NORM_INF) <= 1,
            "GPU grayscale differs from reference"
        );
        auto plan = pack({{8, 4, 28, 32}, {48, 12, 70, 56}}, cw, ch);
        cv::Mat mosaic(plan.mosaic_h, plan.mosaic_w, CV_8UC4, cv::Scalar(114, 114, 114, 255));
        for (const auto& t : plan.tiles)
            image(cv::Rect(t.source.x1, t.source.y1, t.w, t.h))
                .copyTo(mosaic(cv::Rect(t.x, t.y, t.w, t.h)));
        cv::Mat resized, reference(ch, cw, CV_8UC4, cv::Scalar(114, 114, 114, 255));
        cv::resize(mosaic, resized, {plan.resized_w, plan.resized_h}, 0, 0, cv::INTER_LINEAR);
        resized.copyTo(reference(cv::Rect(plan.pad_x, plan.pad_y, plan.resized_w, plan.resized_h)));
        auto rendered = kernels.canvas(view, plan);
        check(
            cv::norm(reference, cv::Mat(ch, cw, CV_8UC4, rendered.data()), cv::NORM_INF) <= 1,
            "GPU canvas differs from reference"
        );
        cuda_check(cudaMalloc(&tensor, 3 * cw * ch * sizeof(float)), "allocate test tensor");
        std::array<float, 3> scales{.01f, .02f, .03f}, offsets{-.5f, .1f, .25f};
        for (bool half : {false, true})
            for (bool bgr : {false, true}) {
                kernels.tensor(view, plan, tensor, half, bgr, scales, offsets);
                std::vector<float> values(3 * cw * ch);
                if (half) {
                    std::vector<__half> halves(values.size());
                    cuda_check(
                        cudaMemcpy(
                            halves.data(),
                            tensor,
                            halves.size() * sizeof(__half),
                            cudaMemcpyDeviceToHost
                        ),
                        "download half tensor"
                    );
                    for (std::size_t i = 0; i < values.size(); ++i)
                        values[i] = __half2float(halves[i]);
                } else
                    cuda_check(
                        cudaMemcpy(
                            values.data(),
                            tensor,
                            values.size() * sizeof(float),
                            cudaMemcpyDeviceToHost
                        ),
                        "download float tensor"
                    );
                cv::Mat expected(ch, cw, CV_8UC4, rendered.data());
                for (int channel = 0; channel < 3; ++channel)
                    for (int y = 0; y < ch; ++y)
                        for (int x = 0; x < cw; ++x) {
                            int source = bgr ? 2 - channel : channel;
                            float want = expected.at<cv::Vec4b>(y, x)[source] * scales[channel] +
                                         offsets[channel];
                            check(
                                std::abs(want - values[channel * cw * ch + y * cw + x]) <
                                    (half ? .008f : 1e-5f),
                                "color/order/normalization mismatch"
                            );
                        }
            }
        for (int i = 0; i < 100; ++i)
            kernels.tensor(view, plan, tensor, false, false, scales, offsets);
        check(kernels.rgba(view) == original, "GPU preprocessing changed source pixels");
        auto* batch = nvds_create_batch_meta(1);
        check(batch != nullptr, "metadata allocation");
        auto* frame = nvds_acquire_frame_meta_from_pool(batch);
        nvds_add_frame_meta_to_batch(batch, frame);
        FramePlan placement;
        placement.component_id = 11;
        placement.status = Status::Active;
        placement.plan = plan;
        placement.tensor_slot = 0;
        attach_plan(batch, frame, placement);
        auto* user = static_cast<NvDsUserMeta*>(frame->frame_user_meta_list->data);
        auto* copied = static_cast<FramePlan*>(user->base_meta.copy_func(user, nullptr));
        check(copied->plan.tiles.size() == plan.tiles.size(), "metadata copy lost tiles");
        copied->plan.tiles.clear();
        check(
            frame_plan(frame, 11)->plan.tiles.size() == plan.tiles.size(),
            "metadata copy shares vector storage"
        );
        NvDsUserMeta wrapper{};
        wrapper.user_meta_data = copied;
        user->base_meta.release_func(&wrapper, nullptr);
        auto* empty = nvds_acquire_user_meta_from_pool(batch);
        empty->base_meta.meta_type = NVDS_PREPROCESS_BATCH_META;
        empty->user_meta_data = new GstNvDsPreProcessBatchMeta{};
        empty->base_meta.release_func = [](gpointer data, gpointer) {
            auto* u = static_cast<NvDsUserMeta*>(data);
            delete static_cast<GstNvDsPreProcessBatchMeta*>(u->user_meta_data);
            u->user_meta_data = nullptr;
        };
        nvds_add_user_meta_to_batch(batch, empty);
        auto* buffer = gst_buffer_new();
        auto* batch_gst = gst_buffer_add_nvds_meta(
            buffer, batch, nullptr, nvds_batch_meta_copy_func, nvds_batch_meta_release_func
        );
        check(batch_gst != nullptr, "GStreamer metadata allocation");
        batch_gst->meta_type = NVDS_BATCH_GST_META;
        GstPadProbeInfo probe{};
        probe.type = GST_PAD_PROBE_TYPE_BUFFER;
        probe.data = buffer;
        check(
            vms_motion_strip_empty_tensors(nullptr, &probe, nullptr) == GST_PAD_PROBE_OK,
            "idle probe dropped video buffer"
        );
        check(
            batch->batch_user_meta_list == nullptr && batch->num_frames_in_batch == 1,
            "idle tensor removal lost original frame"
        );
        gst_buffer_unref(buffer); // Batch ownership transferred to the GstBuffer.
        CustomInitParams init{};
        init.unique_id = 13;
        init.tensor_params.network_input_order = NvDsPreProcessNetworkInputOrder_kNCHW;
        init.tensor_params.network_input_shape = {8, 3, 64, 64};
        init.tensor_params.network_color_format = NvDsPreProcessFormat_RGB;
        init.tensor_params.data_type = NvDsDataType_FP32;
        init.user_configs["gpu-id"] = std::to_string(gpu);
        auto invalid = init;
        invalid.user_configs["periodic-full-frame"] = "1";
        check(initLib(invalid) == nullptr, "unsupported full-frame policy accepted");
        auto* warmup = initLib(init);
        check(warmup != nullptr, "context warmup failed");
        deInitLib(warmup);
        std::size_t free_before = 0, total = 0, free_after = 0;
        cuda_check(
            cudaMemGetInfo(&free_before, &total), "measure GPU memory before lifecycle checks"
        );
        for (int i = 0; i < 100; ++i) {
            auto* ctx = initLib(init);
            check(ctx != nullptr, "context init failed");
            if (i == 0)
                check(initLib(init) == nullptr, "duplicate component ID accepted");
            check(vms_motion_reset_source(13, 0), "live context reset failed");
            deInitLib(ctx);
            check(!vms_motion_reset_source(13, 0), "destroyed context remained in registry");
        }
        cuda_check(cudaDeviceSynchronize(), "finish lifecycle checks");
        cuda_check(
            cudaMemGetInfo(&free_after, &total), "measure GPU memory after lifecycle checks"
        );
        std::cout << "GPU free bytes before/after 100 contexts: " << free_before << " / "
                  << free_after << '\n';
        cudaFree(tensor);
        tensor = nullptr;
        cudaFree(device);
        device = nullptr;
        std::cout << "GPU tensor and metadata checks passed\n";
    } catch (const std::exception& e) {
        cudaFree(tensor);
        cudaFree(device);
        std::cerr << e.what() << '\n';
        return 1;
    }
}
