#include "vms_motion/regions.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace vms_motion {
static double iou(const Box& a, const Box& b) {
    double iw = std::max(0, std::min(a.x2, b.x2) - std::max(a.x1, b.x1)),
           ih = std::max(0, std::min(a.y2, b.y2) - std::max(a.y1, b.y1)), inter = iw * ih,
           area = double(a.width()) * a.height() + double(b.width()) * b.height() - inter;
    return area > 0 ? inter / area : 0;
}

static bool unit(double v) {
    return std::isfinite(v) && v >= 0 && v <= 1;
}

RegionTracker::RegionTracker(RegionConfig c)
    : config_(c) {
    if (c.max_misses < 1 || c.max_regions < 0 || !std::isfinite(c.padding) || c.padding < 0 ||
        !unit(c.match_iou) || !unit(c.region_score) || !unit(c.seed_score))
        throw std::invalid_argument("invalid region configuration");
}

// Best score first, each person takes the unmatched region it overlaps most (IoU >= match_iou, or its centre inside);
// a person matching none opens a new region while there is room. Returns which regions were hit.
std::vector<bool> RegionTracker::match(
    std::uint64_t frame, const std::vector<Detection>& persons, double min_score
) {
    std::vector<bool> hit(regions_.size(), false);
    std::vector<std::size_t> order(persons.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        return persons[a].score > persons[b].score;
    });
    for (auto i : order) {
        const auto& d = persons[i];
        if (d.score < min_score || d.box.width() < 2 || d.box.height() < 2)
            continue;
        double cx = (d.box.x1 + d.box.x2) / 2., cy = (d.box.y1 + d.box.y2) / 2., best_iou = -1;
        int best = -1;
        for (std::size_t r = 0; r < regions_.size(); ++r) {
            const auto& b = regions_[r].box;
            double v = iou(d.box, b);
            bool inside = cx >= b.x1 && cx < b.x2 && cy >= b.y1 && cy < b.y2;
            if (!hit[r] && (v >= config_.match_iou || inside) && v > best_iou) {
                best = int(r);
                best_iou = v;
            }
        }
        if (best >= 0) {
            regions_[best].box = d.box;
            regions_[best].misses = 0;
            regions_[best].last_hit = frame;
            hit[best] = true;
        } else if (int(regions_.size()) < config_.max_regions) {
            regions_.push_back({d.box, 0, frame, frame});
            hit.push_back(true);
        }
    }
    return hit;
}

void RegionTracker::update_mosaic(std::uint64_t frame, const std::vector<Detection>& persons) {
    auto hit = match(frame, persons, config_.region_score);
    std::size_t kept = 0;
    for (std::size_t r = 0; r < regions_.size(); ++r)
        if (hit[r] || ++regions_[r].misses < config_.max_misses)
            regions_[kept++] = regions_[r];
    regions_.resize(kept);
}

void RegionTracker::seed(std::uint64_t frame, const std::vector<Detection>& persons) {
    match(frame, persons, config_.seed_score);
}

std::vector<Box> RegionTracker::crops(int width, int height) const {
    std::vector<Box> out;
    for (const auto& r : regions_) {
        auto b = expand(r.box, config_.padding, width, height);
        if (b.width() >= 2 && b.height() >= 2)
            out.push_back(b);
    }
    return out;
}
}
