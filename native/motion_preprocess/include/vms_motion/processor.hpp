#pragma once

#include "motion.hpp"
#include "regions.hpp"

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
    int full_interval = 0;     // N: a full-frame slot every N frames per source; 0 = never
    bool full_stagger = true;  // offset each source by source_id % N so sources take turns
    RegionConfig regions;
};

struct FramePlan {
    unsigned version = 2, component_id = 0, source_id = 0;
    std::uint64_t frame_num = 0, pts = 0;
    int source_w = 0, source_h = 0;
    Status status = Status::Calibrating;  // of the mosaic; Active means `plan` holds crops
    int tensor_slot = -1;                 // mosaic slot in the tensor, -1 when skipped
    Plan plan;                            // mosaic placements
    int full_slot = -1;                   // full-frame slot in the tensor, -1 when not due
    Plan full_plan;                       // one tile covering the whole frame: the plain letterbox
    int motion_boxes = 0, regions = 0;    // crop sources before merging, for logs
    double motion_ms = 0, packing_ms = 0, tensor_ms = 0;
};

class Processor {
public:
    explicit Processor(Config config);
    // Motion boxes plus the source's person regions -> mosaic plan; also the full-frame plan when it is due.
    FramePlan process(
        unsigned source,
        std::uint64_t frame,
        std::uint64_t pts,
        int width,
        int height,
        const cv::Mat& small_gray
    );
    // Detector feedback for one frame of a source; `mosaic_inferred` says whether its mosaic slot ran.
    void feedback(
        unsigned source,
        std::uint64_t frame,
        bool mosaic_inferred,
        const std::vector<Detection>& mosaic_persons,
        const std::vector<Detection>& full_persons
    );
    bool full_frame_due(unsigned source, std::uint64_t frame) const;

    const Config& config() const {
        return config_;
    }

    void reset_source(unsigned source) {
        sources_.erase(source);
    }

    void reset_all() {
        sources_.clear();
    }

private:
    struct State {
        std::unique_ptr<MotionDetector> motion;
        std::unique_ptr<RegionTracker> regions;
        int width = 0, height = 0;
        std::uint64_t frame = 0, pts = 0;
    };

    Config config_;
    std::unordered_map<unsigned, State> sources_;
};
}
