#include "vms_motion/geometry.hpp"

#include <iomanip>
#include <iostream>

using namespace vms_motion;

int main() {
    int w, h, cw, ch, gap, n;
    double padding, zoom, growth;
    std::cout << std::setprecision(17);
    while (std::cin >> w >> h >> cw >> ch >> padding >> gap >> zoom >> growth >> n) {
        std::vector<Box> boxes;
        for (int i = 0; i < n; ++i) {
            Box b;
            std::cin >> b.x1 >> b.y1 >> b.x2 >> b.y2;
            boxes.push_back(expand(b, padding, w, h));
        }
        auto p = pack(merge(std::move(boxes), growth), cw, ch, gap, zoom);
        std::cout << p.mosaic_w << ' ' << p.mosaic_h << ' ' << p.resized_w << ' ' << p.resized_h
                  << ' ' << p.pad_x << ' ' << p.pad_y << ' ' << p.scale_x << ' ' << p.scale_y << ' '
                  << p.tiles.size();
        for (auto t : p.tiles)
            std::cout << ' ' << t.source.x1 << ' ' << t.source.y1 << ' ' << t.source.x2 << ' '
                      << t.source.y2 << ' ' << t.x << ' ' << t.y << ' ' << t.w << ' ' << t.h;
        std::cout << '\n';
    }
}
