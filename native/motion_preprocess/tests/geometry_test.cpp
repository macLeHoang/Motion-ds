#include "vms_motion/geometry.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace vms_motion;

static void check(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

int main() {
    try {
        check(expand({100, 100, 200, 300}, .25, 640, 360) == Box{75, 50, 225, 350}, "crop padding");
        check(expand({0, 0, 100, 100}, .5, 640, 360) == Box{0, 0, 150, 150}, "crop boundary");
        auto merged = merge({{0, 0, 10, 10}, {5, 5, 20, 20}, {100, 100, 110, 110}}, 0);
        check(merged.size() == 2 && merged[0] == Box{0, 0, 20, 20}, "overlap merge");
        std::vector<Box> crops;
        for (int i = 0; i < 12; ++i)
            crops.push_back({i * 90, (i % 4) * 150, i * 90 + 60, (i % 4) * 150 + 120});
        auto p = pack(crops, 640, 640);
        check(p.tiles.size() == crops.size(), "lost crop");
        for (std::size_t i = 0; i < p.tiles.size(); ++i) {
            auto a = p.tiles[i];
            check(a.w == a.source.width() && a.h == a.source.height(), "native-scale placement");
            for (std::size_t j = i + 1; j < p.tiles.size(); ++j) {
                auto b = p.tiles[j];
                check(
                    a.x + a.w + 4 <= b.x || b.x + b.w + 4 <= a.x || a.y + a.h + 4 <= b.y ||
                        b.y + b.h + 4 <= a.y,
                    "overlapping tiles"
                );
            }
            std::array<double, 4> canvas{
                p.pad_x + (a.x + 5) * p.scale_x,
                p.pad_y + (a.y + 10) * p.scale_y,
                p.pad_x + (a.x + a.w - 5) * p.scale_x,
                p.pad_y + (a.y + a.h - 10) * p.scale_y
            };
            auto mapped = map_box(p, canvas);
            check(
                mapped && std::abs((*mapped)[0] - a.source.x1 - 5) < 1e-8 &&
                    std::abs((*mapped)[3] - a.source.y2 + 10) < 1e-8,
                "box round trip"
            );
            auto point = map_point(p, a, canvas[0], canvas[1]);
            check(std::abs(point[0] - a.source.x1 - 5) < 1e-8, "keypoint round trip");
        }
        check(!map_box(p, {-100, -100, -90, -90}), "gap detection must be dropped");
        check(pack({}, 640, 640).tiles.empty(), "empty mosaic");
        auto rectangular = pack({{0, 0, 100, 50}}, 1280, 640, 4, 2);
        check(rectangular.scale_x <= 2.01 && rectangular.scale_y <= 2.01, "rectangular zoom limit");
        auto clip = pack({{100, 200, 140, 280}}, 640, 640);
        auto t = clip.tiles[0];
        auto mapped = map_box(
            clip,
            {double(clip.pad_x + t.x * clip.scale_x) - 20,
             double(clip.pad_y + t.y * clip.scale_y) - 20,
             clip.pad_x + (t.x + t.w) * clip.scale_x + 20,
             clip.pad_y + (t.y + t.h) * clip.scale_y + 20}
        );
        check(mapped && (*mapped)[0] == 100 && (*mapped)[3] == 280, "source crop clipping");
        bool threw = false;
        try {
            pack({}, 0, 640);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "invalid canvas accepted");
        std::cout << "geometry checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
