#include "vms_motion/processor.hpp"

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
        cv::Mat bg(144, 256, CV_8UC1);
        for (int y = 0; y < bg.rows; ++y)
            for (int x = 0; x < bg.cols; ++x)
                bg.at<unsigned char>(y, x) = 70 + (x / 8 + y / 8) % 40;
        MotionDetector md(bg.cols, bg.rows);
        for (int i = 0; i < 400; ++i)
            md.detect(bg);
        check(!md.calibrating() && md.detect(bg).empty(), "static scene failed to settle");
        auto moved = bg.clone();
        moved(cv::Rect(100, 40, 24, 50)).setTo(250);
        check(!md.detect(moved).empty(), "moving object lost");
        for (int i = 0; i < 600; ++i)
            md.detect(moved);
        check(md.detect(moved).empty(), "persistent change not absorbed");
        md.reset();
        check(md.calibrating(), "reset must calibrate");
        Config c;
        Processor processor(c);
        cv::Mat small;
        cv::resize(bg, small, {100 * bg.cols / bg.rows, 100}, 0, 0, cv::INTER_NEAREST);
        for (int i = 0; i < 400; ++i) {
            processor.process(0, i, i * 100000000ULL, bg.cols, bg.rows, small);
            processor.process(1, i, i * 100000000ULL, bg.cols, bg.rows, small);
        }
        check(
            processor.process(0, 400, 40000000000ULL, bg.cols, bg.rows, small).status ==
                Status::Idle,
            "idle frame generated sample"
        );
        cv::resize(moved, small, {100 * bg.cols / bg.rows, 100}, 0, 0, cv::INTER_NEAREST);
        auto active = processor.process(0, 401, 40100000000ULL, bg.cols, bg.rows, small);
        check(active.status == Status::Active && !active.plan.tiles.empty(), "active crop missing");
        processor.reset_source(0);
        check(
            processor.process(0, 402, 40200000000ULL, bg.cols, bg.rows, small).status ==
                Status::Calibrating,
            "source reset reused background"
        );
        cv::resize(bg, small, {100 * bg.cols / bg.rows, 100}, 0, 0, cv::INTER_NEAREST);
        check(
            processor.process(1, 401, 40100000000ULL, bg.cols, bg.rows, small).status ==
                Status::Idle,
            "other source reset unexpectedly"
        );
        check(
            processor.process(1, 1, 1, bg.cols, bg.rows, small).status == Status::Calibrating,
            "regression did not reset background"
        );
        for (int i = 2; i < 402; ++i)
            processor.process(1, i, i * 100000000ULL, bg.cols, bg.rows, small);
        check(
            processor.process(1, 402, 0, bg.cols, bg.rows, small).status == Status::Calibrating,
            "timestamp regression did not reset background"
        );
        check(
            processor.process(1, 403, 1, 512, 288, small).status == Status::Calibrating,
            "resolution change did not reset background"
        );
        for (int i = 0; i < 100; ++i) {
            processor.reset_all();
            processor.process(0, i, i, bg.cols, bg.rows, small);
        }
        std::cout << "motion and lifecycle checks passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
