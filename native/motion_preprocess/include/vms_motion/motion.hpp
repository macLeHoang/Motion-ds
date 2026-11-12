#pragma once

#include "geometry.hpp"

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

namespace vms_motion {
using Polygon = std::vector<cv::Point2d>;

struct MotionConfig {
    int height = 100, threshold = 30, contour_area = 10, history = 50, stale_frames = 0;
    double alpha = 0.01, lightning_threshold = 0.8;
    bool improve_contrast = true;
    std::vector<Polygon> masks;
};

class MotionDetector {
public:
    MotionDetector(int width, int height, MotionConfig config = {});
    std::vector<Box> detect(const cv::Mat& native_gray);
    std::vector<Box> detect_small(const cv::Mat& small_gray);
    void reset();
    void protect(const std::vector<Box>& boxes);
    void update_masks(std::vector<Polygon> masks);

    bool calibrating() const {
        return calibrating_;
    }

    int small_width() const {
        return sw_;
    }

    int small_height() const {
        return sh_;
    }

    std::uint64_t rebases() const {
        return rebases_;
    }

private:
    cv::Mat prepare(cv::Mat small, bool update = true);
    void relearn(const cv::Mat& small, const cv::Mat& prepared, const cv::Mat& delta);
    int width_, height_, sw_, sh_, history_index_ = 0, motion_count_ = 0, frames_ = 0;
    MotionConfig config_;
    cv::Mat average_, mask_, previous_, stale_, protected_;
    std::vector<cv::Vec2b> contrast_;
    bool calibrating_ = true;
    std::uint64_t rebases_ = 0;
};
}
