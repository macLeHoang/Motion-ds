#pragma once
// Internal: the state of one nvdspreprocess component, shared by the tensor preparation (preprocess.cpp) and the
// decode probe on the following nvinfer (decoder.cpp).
// Locking: `mutex` guards processor, gpu and preview state and is held by the tensor preparation, also while it waits
// for a free tensor buffer. Tensor buffers come back only after nvinfer's output has passed the decode probe, so the
// probe must never take `mutex`: it only appends to `feedback` (under `feedback_mutex`), which the tensor preparation
// applies at the start of the next batch.

#include "vms_motion/gpu.hpp"
#include "vms_motion/processor.hpp"

#include <array>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>

namespace vms_motion {
struct DecoderConfig {
    unsigned infer_id = 2;            // gie-unique-id of the nvinfer reading this component's tensors
    int num_classes = 81;
    float det_score = .25f, dedupe_iou = .5f;
    std::vector<bool> mosaic_classes;  // kept from mosaic slots; their full-frame boxes seed regions
    std::vector<bool> full_classes;    // kept from full-frame slots
    bool hold_full = false;            // re-attach full-frame-class detections until the next full frame
    std::vector<std::string> labels;
    void* parser_lib = nullptr;        // dlopen handle
    void* parse = nullptr;             // NvDsInferParseCustomFunc-style instance-mask parser, e.g. NvDsInferYoloAttr
};

// A full-frame-class detection from a source's last full frame, re-attached to its frames until the next full frame
// (full-frame-hold): with detections only every N frames, nvtracker would end those tracks before they are confirmed.
struct HeldDetection {
    unsigned class_id = 0, mask_width = 0, mask_height = 0;
    float confidence = 0, box[4]{};
    std::vector<float> payload;
};

// Detector feedback for one frame, queued by the decode probe for Processor::feedback.
struct Feedback {
    unsigned source = 0;
    std::uint64_t frame = 0;
    bool mosaic_ran = false;
    std::vector<Detection> persons, seeds;
};

struct Context {
    unsigned id = 0;
    int device = 0, capacity = 0;
    Config config;
    DecoderConfig decoder;
    bool fp16 = false, bgr = false;
    std::array<float, 3> scale{1.f / 255, 1.f / 255, 1.f / 255}, offset{0, 0, 0};
    std::unique_ptr<Processor> processor;
    std::unique_ptr<Gpu> gpu;
    std::mutex mutex;
    std::mutex feedback_mutex;
    std::vector<Feedback> feedback;
    std::mutex held_mutex;  // decode probe only
    std::unordered_map<unsigned, std::vector<HeldDetection>> held;  // by source id
    std::string preview_dir;
    std::string preview_session;
    double preview_fps = 1;
    std::uint64_t preview_seq = 0;
    std::unordered_map<unsigned, double> last_preview;
    std::ofstream events;

    ~Context() {
        cudaSetDevice(device);
        gpu.reset();
        if (decoder.parser_lib)
            dlclose(decoder.parser_lib);
    }
};

// The live context of a component id, or nullptr.
std::shared_ptr<Context> find_context(unsigned component_id);
}
