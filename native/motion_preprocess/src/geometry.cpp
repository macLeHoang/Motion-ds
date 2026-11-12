#include "vms_motion/geometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace vms_motion {
static double area(Box b) {
    return double(std::max(0, b.width())) * std::max(0, b.height());
}

Box expand(Box b, double p, int w, int h) {
    if (!std::isfinite(p) || p < 0 || w <= 0 || h <= 0)
        throw std::invalid_argument("invalid crop padding/size");
    const double dx = b.width() * p, dy = b.height() * p;
    return {
        int(std::clamp(b.x1 - dx, 0., double(w))),
        int(std::clamp(b.y1 - dy, 0., double(h))),
        int(std::clamp(b.x2 + dx, 0., double(w))),
        int(std::clamp(b.y2 + dy, 0., double(h)))
    };
}

std::vector<Box> merge(std::vector<Box> out, double growth) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < out.size() && !changed; ++i) {
            for (size_t j = i + 1; j < out.size(); ++j) {
                auto a = out[i], b = out[j];
                if (a.x1 > b.x2 || b.x1 > a.x2 || a.y1 > b.y2 || b.y1 > a.y2)
                    continue;
                Box u{
                    std::min(a.x1, b.x1),
                    std::min(a.y1, b.y1),
                    std::max(a.x2, b.x2),
                    std::max(a.y2, b.y2)
                };
                Box intersection{
                    std::max(a.x1, b.x1),
                    std::max(a.y1, b.y1),
                    std::min(a.x2, b.x2),
                    std::min(a.y2, b.y2)
                };
                if (growth > 0 && area(u) > growth * (area(a) + area(b) - area(intersection)))
                    continue;
                out[i] = u;
                out.erase(out.begin() + j);
                changed = true;
                break;
            }
        }
    }
    return out;
}

Plan pack(const std::vector<Box>& boxes, int cw, int ch, int gap, double zoom) {
    if (cw <= 0 || ch <= 0 || gap < 0 || !std::isfinite(zoom) || zoom <= 0)
        throw std::invalid_argument("invalid canvas/gap/zoom");
    Plan out;
    out.canvas_w = cw;
    out.canvas_h = ch;
    std::vector<Box> valid;
    for (auto b : boxes)
        if (b.width() >= 2 && b.height() >= 2)
            valid.push_back(b);
    if (valid.empty())
        return out;
    std::vector<size_t> order(valid.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
        return valid[a].height() > valid[b].height();
    });
    double total = 0;
    int widest = 0;
    for (auto b : valid) {
        total += double(b.width() + gap) * (b.height() + gap);
        widest = std::max(widest, b.width());
    }
    std::tuple<int, long long> best{INT32_MAX, INT64_MAX};
    for (double f : {0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 2.0}) {
        int width = std::max(widest, int(std::sqrt(total) * f));
        int x = 0, y = 0, shelf_h = 0, used_w = 0;
        std::vector<Placement> tiles(valid.size());
        for (auto i : order) {
            auto b = valid[i];
            if (x > 0 && x + b.width() > width) {
                y += shelf_h + gap;
                x = 0;
                shelf_h = 0;
            }
            tiles[i] = {b, x, y, b.width(), b.height()};
            used_w = std::max(used_w, x + b.width());
            x += b.width() + gap;
            shelf_h = std::max(shelf_h, b.height());
        }
        int height = y + shelf_h;
        auto key =
            std::make_tuple(std::max(used_w, height), static_cast<long long>(used_w) * height);
        if (key < best) {
            best = key;
            out.tiles = std::move(tiles);
            out.mosaic_w = used_w;
            out.mosaic_h = height;
        }
    }
    // Pad the shorter canvas-relative image sufficiently to cap magnification.
    const double uncapped = std::min(double(cw) / out.mosaic_w, double(ch) / out.mosaic_h);
    if (uncapped > zoom) {
        if (out.mosaic_h >= out.mosaic_w)
            out.mosaic_h = std::max(out.mosaic_h, int(std::ceil(ch / zoom)));
        else
            out.mosaic_w = std::max(out.mosaic_w, int(std::ceil(cw / zoom)));
    }
    const double scale = std::min(double(cw) / out.mosaic_w, double(ch) / out.mosaic_h);
    out.resized_w = std::max(1, int(std::nearbyint(out.mosaic_w * scale)));
    out.resized_h = std::max(1, int(std::nearbyint(out.mosaic_h * scale)));
    out.pad_x = (cw - out.resized_w) / 2;
    out.pad_y = (ch - out.resized_h) / 2;
    out.scale_x = double(out.resized_w) / out.mosaic_w;
    out.scale_y = double(out.resized_h) / out.mosaic_h;
    return out;
}

std::array<double, 2> map_point(const Plan& p, const Placement& t, double x, double y) {
    if (p.scale_x <= 0 || p.scale_y <= 0)
        throw std::invalid_argument("empty placement plan");
    return {
        (x - p.pad_x) / p.scale_x - t.x + t.source.x1, (y - p.pad_y) / p.scale_y - t.y + t.source.y1
    };
}

std::optional<std::array<double, 4>> map_box(const Plan& p, const std::array<double, 4>& b) {
    if (p.tiles.empty() || p.scale_x <= 0 || p.scale_y <= 0)
        return std::nullopt;
    for (auto v : b)
        if (!std::isfinite(v))
            return std::nullopt;
    if (b[2] <= b[0] || b[3] <= b[1])
        return std::nullopt;
    double cx = ((b[0] + b[2]) / 2 - p.pad_x) / p.scale_x,
           cy = ((b[1] + b[3]) / 2 - p.pad_y) / p.scale_y;
    for (const auto& t : p.tiles) {
        if (cx < t.x || cx >= t.x + t.w || cy < t.y || cy >= t.y + t.h)
            continue;
        auto lo = map_point(p, t, b[0], b[1]), hi = map_point(p, t, b[2], b[3]);
        return std::array<double, 4>{
            std::clamp(lo[0], double(t.source.x1), double(t.source.x2)),
            std::clamp(lo[1], double(t.source.y1), double(t.source.y2)),
            std::clamp(hi[0], double(t.source.x1), double(t.source.x2)),
            std::clamp(hi[1], double(t.source.y1), double(t.source.y2))
        };
    }
    return std::nullopt;
}
}
