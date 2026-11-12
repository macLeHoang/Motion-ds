#pragma once
// Internal: the state of one nvdspreprocess component, shared by the tensor preparation (preprocess.cpp) and the
// decode probe on the following nvinfer (decoder.cpp). Lock `mutex` before touching `processor` or `gpu`.

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
    std::vector<std::string> labels;
    void* parser_lib = nullptr;        // dlopen handle
    void* parse = nullptr;             // NvDsInferParseCustomFunc-style instance-mask parser, e.g. NvDsInferYoloAttr
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
