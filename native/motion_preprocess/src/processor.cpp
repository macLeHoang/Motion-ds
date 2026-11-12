#include "vms_motion/processor.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>

namespace vms_motion {
using Clock = std::chrono::steady_clock;

static double elapsed(Clock::time_point a) {
    return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}

Processor::Processor(Config config)
    : config_(std::move(config)) {
    if (!std::isfinite(config_.padding) || config_.padding < 0 ||
        !std::isfinite(config_.merge_growth) || config_.merge_growth < 0)
        throw std::invalid_argument("invalid padding/merge growth");
    if (config_.full_interval < 0)
        throw std::invalid_argument("invalid full-frame interval");
    pack({}, config_.canvas_w, config_.canvas_h, config_.gap, config_.max_zoom);
    MotionDetector check(640, 360, config_.motion);
    RegionTracker regions(config_.regions);
}

bool Processor::full_frame_due(unsigned source, std::uint64_t frame) const {
    if (config_.full_interval <= 0)
        return false;
    auto n = std::uint64_t(config_.full_interval);
    return (frame + (config_.full_stagger ? source % n : 0)) % n == 0;
}

void Processor::feedback(
    unsigned source,
    std::uint64_t frame,
    bool mosaic_inferred,
    const std::vector<Detection>& mosaic_persons,
    const std::vector<Detection>& full_persons
) {
    auto it = sources_.find(source);
    if (it == sources_.end() || !it->second.regions)
        return;  // source was reset since that frame
    if (mosaic_inferred)
        it->second.regions->update_mosaic(frame, mosaic_persons);
    it->second.regions->seed(frame, full_persons);
}

FramePlan Processor::process(
    unsigned source, std::uint64_t frame, std::uint64_t pts, int w, int h, const cv::Mat& small
) {
    auto& s = sources_[source];
    const auto no_pts = UINT64_MAX;
    if (!s.motion || s.width != w || s.height != h || frame < s.frame ||
        (pts != no_pts && s.pts != no_pts && pts < s.pts)) {
        auto motion_config = config_.motion;
        auto masks = config_.source_masks.find(source);
        if (masks != config_.source_masks.end())
            motion_config.masks = masks->second;
        s.motion = std::make_unique<MotionDetector>(w, h, std::move(motion_config));
        s.regions = std::make_unique<RegionTracker>(config_.regions);
        s.width = w;
        s.height = h;
    }
    s.frame = frame;
    s.pts = pts;
    FramePlan out;
    out.plan.canvas_w = config_.canvas_w;
    out.plan.canvas_h = config_.canvas_h;
    out.source_id = source;
    out.frame_num = frame;
    out.pts = pts;
    out.source_w = w;
    out.source_h = h;
    auto start = Clock::now();
    auto boxes = s.motion->detect_small(small);
    out.motion_ms = elapsed(start);
    const bool calibrating = s.motion->calibrating();
    start = Clock::now();
    if (calibrating)
        boxes.clear();  // regions still crop while the motion background settles
    for (auto& b : boxes)
        b = expand(b, config_.padding, w, h);
    auto regions = s.regions->crops(w, h);
    out.motion_boxes = int(boxes.size());
    out.regions = int(regions.size());
    boxes.insert(boxes.end(), regions.begin(), regions.end());
    out.plan = pack(
        merge(std::move(boxes), config_.merge_growth),
        config_.canvas_w,
        config_.canvas_h,
        config_.gap,
        config_.max_zoom
    );
    if (full_frame_due(source, frame))
        out.full_plan = pack({{0, 0, w, h}}, config_.canvas_w, config_.canvas_h, config_.gap, config_.max_zoom);
    out.packing_ms = elapsed(start);
    out.status = !out.plan.tiles.empty() ? Status::Active
                 : calibrating           ? Status::Calibrating
                                         : Status::Idle;
    return out;
}
}
