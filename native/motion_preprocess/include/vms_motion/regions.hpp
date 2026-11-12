#pragma once

#include "geometry.hpp"

#include <cstdint>
#include <vector>

namespace vms_motion {
// Person regions kept from detector feedback, so people who stop moving stay in the mosaic.
struct RegionConfig {
    int max_misses = 25;          // mosaic inferences without a person inside before a region is removed
    double padding = .2;          // growth of a region box when it becomes a crop
    double match_iou = .3;        // detection <-> region match (a detection centre inside also matches)
    double region_score = .5;     // mosaic person score that refreshes or creates a region
    double seed_score = .5;       // full-frame person score that refreshes or creates a region
    int max_regions = 16;         // per source
};

struct Detection {
    Box box;
    float score = 0;
};

struct Region {
    Box box;                      // last matched detection, frame pixels
    int misses = 0;
    std::uint64_t created = 0, last_hit = 0;
};

class RegionTracker {
public:
    explicit RegionTracker(RegionConfig config = {});
    // Persons found in one inferred mosaic: refresh or create regions; every region without a hit misses once.
    void update_mosaic(std::uint64_t frame, const std::vector<Detection>& persons);
    // Persons found in a full frame: refresh or create regions, never count misses.
    void seed(std::uint64_t frame, const std::vector<Detection>& persons);
    // Region boxes grown by the padding and clipped to the frame, ready to merge with motion boxes.
    std::vector<Box> crops(int width, int height) const;

    const std::vector<Region>& regions() const {
        return regions_;
    }

    void clear() {
        regions_.clear();
    }

private:
    std::vector<bool> match(std::uint64_t frame, const std::vector<Detection>& persons, double min_score);
    RegionConfig config_;
    std::vector<Region> regions_;
};
}
