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
    pack({}, config_.canvas_w, config_.canvas_h, config_.gap, config_.max_zoom);
    MotionDetector check(640, 360, config_.motion);
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
    if (s.motion->calibrating())
        return out;
    start = Clock::now();
    for (auto& b : boxes)
        b = expand(b, config_.padding, w, h);
    out.plan = pack(
        merge(std::move(boxes), config_.merge_growth),
        config_.canvas_w,
        config_.canvas_h,
        config_.gap,
        config_.max_zoom
    );
    out.packing_ms = elapsed(start);
    out.status = out.plan.tiles.empty() ? Status::Idle : Status::Active;
    return out;
}
}
