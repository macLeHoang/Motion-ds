#pragma once

#include "motion.hpp"

#include <memory>
#include <unordered_map>

namespace vms_motion {
enum class Status {
    Calibrating,
    Idle,
    Active
};

inline const char* status_name(Status s) {
    return s == Status::Active ? "active" : s == Status::Idle ? "idle" : "calibrating";
}

struct Config {
    MotionConfig motion;
    int canvas_w = 640, canvas_h = 640, gap = 4;
    double padding = .15, max_zoom = 2, merge_growth = 1.3;
    std::unordered_map<unsigned, std::vector<Polygon>> source_masks;
};

struct FramePlan {
    unsigned version = 1, component_id = 0, source_id = 0;
    std::uint64_t frame_num = 0, pts = 0;
    int source_w = 0, source_h = 0, tensor_slot = -1;
    Status status = Status::Calibrating;
    Plan plan;
    double motion_ms = 0, packing_ms = 0, tensor_ms = 0;
};

class Processor {
public:
    explicit Processor(Config config);
    FramePlan process(
        unsigned source,
        std::uint64_t frame,
        std::uint64_t pts,
        int width,
        int height,
        const cv::Mat& small_gray
    );

    void reset_source(unsigned source) {
        sources_.erase(source);
    }

    void reset_all() {
        sources_.clear();
    }

private:
    struct State {
        std::unique_ptr<MotionDetector> motion;
        int width = 0, height = 0;
        std::uint64_t frame = 0, pts = 0;
    };

    Config config_;
    std::unordered_map<unsigned, State> sources_;
};
}
