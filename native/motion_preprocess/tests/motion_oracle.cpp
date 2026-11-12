#include "vms_motion/motion.hpp"

#include <iostream>

using namespace vms_motion;

int main() {
    int w, h, polys, protected_count;
    MotionConfig c;
    if (!(std::cin >> w >> h >> c.height >> c.threshold >> c.stale_frames >> c.improve_contrast >>
          polys))
        return 1;
    for (int i = 0; i < polys; ++i) {
        int n;
        std::cin >> n;
        Polygon p;
        for (int j = 0; j < n; ++j) {
            double x, y;
            std::cin >> x >> y;
            p.emplace_back(x, y);
        }
        c.masks.push_back(p);
    }
    std::cin >> protected_count;
    std::vector<Box> protected_boxes;
    for (int i = 0; i < protected_count; ++i) {
        Box b;
        std::cin >> b.x1 >> b.y1 >> b.x2 >> b.y2;
        protected_boxes.push_back(b);
    }
    if (std::cin.get() != '\n')
        return 2;
    MotionDetector md(w, h, c);
    md.protect(protected_boxes);
    cv::Mat image(h, w, CV_8UC1);
    while (std::cin.read(reinterpret_cast<char*>(image.data), w * h)) {
        auto boxes = md.detect(image);
        std::cout << md.calibrating() << ' ' << boxes.size();
        for (auto b : boxes)
            std::cout << ' ' << b.x1 << ' ' << b.y1 << ' ' << b.x2 << ' ' << b.y2;
        std::cout << '\n';
    }
}
