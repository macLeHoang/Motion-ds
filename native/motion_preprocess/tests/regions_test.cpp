#include "vms_motion/processor.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

using namespace vms_motion;

static void check(bool b, const char* m) {
    if (!b)
        throw std::runtime_error(m);
}

int main() {
    try {
        RegionConfig rc;
        rc.max_misses = 3;
        rc.max_regions = 2;
        RegionTracker t(rc);
        t.update_mosaic(1, {{{100, 100, 140, 220}, .9f}});
        check(t.regions().size() == 1, "person did not open a region");
        t.update_mosaic(2, {{{104, 102, 146, 224}, .8f}});
        check(t.regions().size() == 1 && t.regions()[0].box.x1 == 104 && t.regions()[0].misses == 0,
              "moved person did not refresh its region");
        t.update_mosaic(3, {{{100, 100, 140, 220}, .3f}});
        check(t.regions()[0].misses == 1, "person below region-score refreshed a region");
        t.update_mosaic(4, {});
        check(t.regions().size() == 1, "region removed before K misses");
        t.update_mosaic(5, {});
        check(t.regions().empty(), "region kept after K misses");
        t.seed(6, {{{10, 10, 50, 90}, .9f}});
        check(t.regions().size() == 1, "full-frame person did not seed a region");
        for (int i = 7; i < 12; ++i)
            t.seed(i, {});
        check(t.regions().size() == 1 && t.regions()[0].misses == 0, "full-frame feedback counted misses");
        t.update_mosaic(12, {{{10, 10, 50, 90}, .9f}, {{300, 10, 340, 90}, .8f}, {{500, 10, 540, 90}, .7f}});
        check(t.regions().size() == 2, "max-regions not enforced");
        auto crops = t.crops(400, 300);  // 40x80 boxes grown by 20 %: 8 px sideways, 16 px up/down, clipped
        check(crops.size() == 2 && crops[0] == Box{2, 0, 58, 106} && crops[1] == Box{292, 0, 348, 106},
              "region crops not padded and clipped");
        bool threw = false;
        try {
            RegionConfig bad;
            bad.max_misses = 0;
            RegionTracker invalid(bad);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        check(threw, "region-max-misses 0 accepted");

        Config c;
        c.canvas_w = c.canvas_h = 768;
        c.full_interval = 25;
        c.regions.max_misses = 3;
        Processor p(c);
        check(p.full_frame_due(0, 0) && p.full_frame_due(0, 25) && !p.full_frame_due(0, 1), "source 0 schedule");
        check(p.full_frame_due(1, 24) && p.full_frame_due(1, 49) && !p.full_frame_due(1, 0), "sources not staggered");
        auto no_stagger = c;
        no_stagger.full_stagger = false;
        check(Processor(no_stagger).full_frame_due(1, 0), "full-frame-stagger 0 still offsets sources");
        auto never = c;
        never.full_interval = 0;
        check(!Processor(never).full_frame_due(0, 0), "full-frame-interval 0 must disable full frames");

        cv::Mat bg(720, 1280, CV_8UC1), small;
        for (int y = 0; y < bg.rows; ++y)
            for (int x = 0; x < bg.cols; ++x)
                bg.at<unsigned char>(y, x) = 70 + (x / 40 + y / 40) % 40;
        cv::resize(bg, small, {100 * bg.cols / bg.rows, 100}, 0, 0, cv::INTER_NEAREST);
        auto step = [&](std::uint64_t frame) { return p.process(0, frame, frame * 40000000ULL, 1280, 720, small); };
        FramePlan f;
        for (std::uint64_t i = 0; i < 100; ++i)
            f = step(i);
        check(f.status == Status::Idle && f.plan.tiles.empty() && f.full_plan.tiles.empty(), "static frame not idle");
        f = step(100);
        const auto& full = f.full_plan;
        check(full.tiles.size() == 1 && std::abs(full.scale_x - .6) < 1e-9 && full.resized_w == 768 &&
                  full.resized_h == 432 && full.pad_x == 0 && full.pad_y == 168,
              "full-frame plan is not the Ultralytics letterbox");
        p.feedback(0, 100, false, {}, {{{600, 300, 680, 560}, .9f}});
        f = step(101);
        check(f.status == Status::Active && f.regions == 1 && f.motion_boxes == 0 && !f.plan.tiles.empty(),
              "seeded region not in the mosaic of a static frame");
        for (std::uint64_t i = 102; i < 105; ++i) {
            p.feedback(0, i - 1, true, {}, {});
            f = step(i);
        }
        check(f.status == Status::Idle && f.regions == 0, "region kept after K empty mosaics");
        p.feedback(0, 104, false, {}, {{{600, 300, 680, 560}, .9f}});
        p.reset_source(0);
        check(step(105).regions == 0, "source reset kept regions");
        std::cout << "region and full-frame checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
