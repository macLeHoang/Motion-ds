// Algorithm derived from dls/core/motion.py and Frigate (MIT).
// Copyright (c) 2020 Blake Blackshear. See THIRD_PARTY_NOTICES.md.
#include "vms_motion/motion.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace vms_motion {
MotionDetector::MotionDetector(int w, int h, MotionConfig c)
    : width_(w),
      height_(h),
      config_(std::move(c)) {
    if (w <= 0 || h <= 0 || config_.height <= 0 || config_.history <= 0 || config_.threshold < 1 ||
        config_.threshold > 255 || config_.contour_area < 0 || config_.stale_frames < 0 ||
        !std::isfinite(config_.alpha) || config_.alpha <= 0 || config_.alpha > 1 ||
        !std::isfinite(config_.lightning_threshold) || config_.lightning_threshold <= 0 ||
        config_.lightning_threshold > 1)
        throw std::invalid_argument("invalid motion configuration");
    sh_ = config_.height;
    sw_ = std::max(1, int(static_cast<long long>(sh_) * w / h));
    reset();
}

void MotionDetector::reset() {
    average_ = cv::Mat::zeros(sh_, sw_, CV_32F);
    previous_.release();
    stale_ = cv::Mat::zeros(sh_, sw_, CV_16U);
    protected_ = cv::Mat::zeros(sh_, sw_, CV_8U);
    mask_ = cv::Mat::zeros(sh_, sw_, CV_8U);
    for (const auto& poly : config_.masks) {
        std::vector<cv::Point> pts;
        for (auto p : poly) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 0 || p.x > 1 || p.y < 0 ||
                p.y > 1)
                throw std::invalid_argument("mask coordinates must be normalized to [0,1]");
            pts.emplace_back(int(std::nearbyint(p.x * sw_)), int(std::nearbyint(p.y * sh_)));
        }
        if (pts.size() >= 3)
            cv::fillPoly(mask_, std::vector<std::vector<cv::Point>>{pts}, cv::Scalar(255));
    }
    contrast_.assign(config_.history, cv::Vec2b(0, 255));
    history_index_ = 0;
    calibrating_ = true;
    motion_count_ = 0;
    frames_ = 0;
    observed_ = 0;
    rebases_ = 0;
}

void MotionDetector::update_masks(std::vector<Polygon> masks) {
    config_.masks = std::move(masks);
    reset();
}

void MotionDetector::protect(const std::vector<Box>& boxes) {
    protected_.setTo(0);
    double f = double(height_) / sh_;
    for (auto b : boxes) {
        int x1 = std::clamp(int(b.x1 / f), 0, sw_), y1 = std::clamp(int(b.y1 / f), 0, sh_);
        int x2 = std::clamp(int(std::ceil(b.x2 / f)) + 1, 0, sw_),
            y2 = std::clamp(int(std::ceil(b.y2 / f)) + 1, 0, sh_);
        if (x2 > x1 && y2 > y1)
            protected_(cv::Rect(x1, y1, x2 - x1, y2 - y1)).setTo(255);
    }
}

static int percentile(const std::array<int, 256>& hist, int n, double fraction) {
    double at = (n - 1) * fraction;
    int lo = int(std::floor(at)), hi = int(std::ceil(at)), a = 0, b = 0, sum = 0;
    bool found = false;
    for (int i = 0; i < 256; ++i) {
        sum += hist[i];
        if (!found && sum > lo) {
            a = i;
            found = true;
        }
        if (sum > hi) {
            b = i;
            break;
        }
    }
    return int(a + (b - a) * (at - lo));
}

cv::Mat MotionDetector::prepare(cv::Mat small, bool update) {
    if (config_.improve_contrast) {
        std::array<int, 256> hist{};
        for (int y = 0; y < sh_; ++y)
            for (int x = 0; x < sw_; ++x)
                ++hist[small.at<unsigned char>(y, x)];
        int lo = percentile(hist, sw_ * sh_, .04), hi = percentile(hist, sw_ * sh_, .96);
        if (lo < hi) {
            if (update) {
                contrast_[history_index_] = {
                    static_cast<unsigned char>(lo), static_cast<unsigned char>(hi)
                };
                history_index_ = (history_index_ + 1) % config_.history;
            }
            double mn = 0, mx = 0;
            for (auto v : contrast_) {
                mn += v[0];
                mx += v[1];
            }
            mn /= config_.history;
            mx /= config_.history;
            for (int y = 0; y < sh_; ++y)
                for (int x = 0; x < sw_; ++x) {
                    auto& v = small.at<unsigned char>(y, x);
                    v = static_cast<unsigned char>(
                        (std::clamp(double(v), mn, mx) - mn) / (mx - mn) * 255
                    );
                }
        }
    }
    small.setTo(0, mask_);
    cv::Mat out;
    cv::GaussianBlur(small, out, cv::Size(3, 3), 1);
    return out;
}

void MotionDetector::relearn(const cv::Mat& small, const cv::Mat& image, const cv::Mat& delta) {
    ++frames_;
    int half = std::max(1, config_.threshold / 2), count = 0;
    cv::Mat snap = cv::Mat::zeros(sh_, sw_, CV_8U);
    for (int y = 0; y < sh_; ++y)
        for (int x = 0; x < sw_; ++x) {
            bool changed =
                delta.at<unsigned char>(y, x) > half && !protected_.at<unsigned char>(y, x);
            bool still = !previous_.empty() && std::abs(
                                                   int(image.at<unsigned char>(y, x)) -
                                                   int(previous_.at<unsigned char>(y, x))
                                               ) < half;
            auto& v = stale_.at<unsigned short>(y, x);
            v = changed && still ? std::min(int(v) + 1, 60000) : 0;
            if (v >= config_.stale_frames) {
                snap.at<unsigned char>(y, x) = 255;
                ++count;
            }
        }
    previous_ = image.clone();
    if (double(count) / (sw_ * sh_) >= .15 && frames_ > config_.history) {
        auto last = contrast_[(history_index_ + config_.history - 1) % config_.history];
        std::fill(contrast_.begin(), contrast_.end(), last);
        auto rebased = prepare(small.clone(), false);
        rebased.convertTo(average_, CV_32F);
        previous_ = rebased;
        stale_.setTo(0);
        ++rebases_;
    } else if (count) {
        cv::Mat f;
        image.convertTo(f, CV_32F);
        f.copyTo(average_, snap);
        stale_.setTo(0, snap);
    }
}

std::vector<Box> MotionDetector::detect(const cv::Mat& gray) {
    if (gray.type() != CV_8UC1 || gray.cols != width_ || gray.rows != height_)
        throw std::invalid_argument("grayscale frame shape mismatch");
    cv::Mat small;
    cv::resize(gray, small, cv::Size(sw_, sh_), 0, 0, cv::INTER_NEAREST);
    return detect_small(small);
}

std::vector<Box> MotionDetector::detect_small(const cv::Mat& small) {
    if (small.type() != CV_8UC1 || small.cols != sw_ || small.rows != sh_)
        throw std::invalid_argument("motion image shape mismatch");
    cv::Mat image = prepare(small.clone()), avg, delta, binary;
    cv::convertScaleAbs(average_, avg);
    cv::absdiff(image, avg, delta);
    if (config_.stale_frames)
        relearn(small, image, delta);
    cv::threshold(delta, binary, config_.threshold, 255, cv::THRESH_BINARY);
    cv::dilate(binary, binary, cv::Mat(), cv::Point(-1, -1), 1);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<Box> boxes;
    double total = 0, f = double(height_) / sh_;
    for (const auto& contour : contours) {
        double a = cv::contourArea(contour);
        total += a;
        if (a > config_.contour_area) {
            auto r = cv::boundingRect(contour);
            boxes.push_back(
                {int(r.x * f), int(r.y * f), int((r.x + r.width) * f), int((r.y + r.height) * f)}
            );
        }
    }
    double pct = total / (sw_ * sh_);
    // The contrast history starts at (0, 255) and keeps changing the normalised image until it is full, which reads as
    // motion on a static scene; so calibration ends only once every history entry comes from real frames.
    ++observed_;
    bool settled = !config_.improve_contrast || observed_ >= config_.history;
    if (pct < .05 && boxes.size() <= 4 && settled)
        calibrating_ = false;
    if (calibrating_ || pct > config_.lightning_threshold)
        calibrating_ = true;
    double alpha = calibrating_ ? .2 : config_.alpha;
    bool absorb = false;
    if (!boxes.empty())
        absorb = ++motion_count_ >= 10;
    else {
        absorb = true;
        motion_count_ = 0;
    }
    if (absorb) {
        cv::Mat floating;
        image.convertTo(floating, CV_32F);
        cv::accumulateWeighted(floating, average_, alpha);
    }
    return boxes;
}
}
