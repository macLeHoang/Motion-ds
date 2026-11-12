#pragma once

#include <array>
#include <optional>
#include <vector>

namespace vms_motion {
struct Box {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    int width() const {
        return x2 - x1;
    }

    int height() const {
        return y2 - y1;
    }

    bool operator==(const Box& b) const {
        return x1 == b.x1 && y1 == b.y1 && x2 == b.x2 && y2 == b.y2;
    }
};

struct Placement {
    Box source;
    int x = 0, y = 0, w = 0, h = 0;
};

struct Plan {
    int canvas_w = 0, canvas_h = 0;
    int mosaic_w = 0, mosaic_h = 0;
    int resized_w = 0, resized_h = 0, pad_x = 0, pad_y = 0;
    double scale_x = 0, scale_y = 0;
    std::vector<Placement> tiles;
};

Box expand(Box b, double padding, int width, int height);
std::vector<Box> merge(std::vector<Box> boxes, double max_growth = 1.3);
Plan pack(
    const std::vector<Box>& boxes, int canvas_w, int canvas_h, int gap = 4, double max_zoom = 2.0
);
// Incoming boxes/points are in final model-input coordinates, before any SDK ROI remapping.
std::optional<std::array<double, 4>> map_box(const Plan&, const std::array<double, 4>&);
std::array<double, 2> map_point(const Plan&, const Placement&, double x, double y);
}
