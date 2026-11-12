#include "context.hpp"
#include "vms_motion/metadata.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_set>

#include <nlohmann/json.hpp>
#include <nvdspreprocess_interface.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace vms_motion;
using Json = nlohmann::json;

namespace {
std::mutex registry_mutex;
std::unordered_map<unsigned, std::weak_ptr<Context>> registry;

double number(
    const std::unordered_map<std::string, std::string>& values, const char* key, double fallback
) {
    auto it = values.find(key);
    if (it == values.end())
        return fallback;
    std::size_t end = 0;
    double v = std::stod(it->second, &end);
    if (end != it->second.size() || !std::isfinite(v))
        throw std::invalid_argument(std::string("invalid ") + key);
    return v;
}

int integer(
    const std::unordered_map<std::string, std::string>& values, const char* key, int fallback
) {
    double v = number(values, key, fallback);
    if (std::floor(v) != v || v < INT32_MIN || v > INT32_MAX)
        throw std::invalid_argument(std::string("invalid integer ") + key);
    return int(v);
}

std::array<float, 3> triple(
    const std::unordered_map<std::string, std::string>& values,
    const char* key,
    std::array<float, 3> fallback
) {
    auto it = values.find(key);
    if (it == values.end())
        return fallback;
    std::istringstream input(it->second);
    std::string token;
    for (auto& value : fallback) {
        if (!std::getline(input, token, ';'))
            throw std::invalid_argument(std::string(key) + " requires three values");
        std::size_t end = 0;
        value = std::stof(token, &end);
        if (end != token.size() || !std::isfinite(value))
            throw std::invalid_argument(std::string("invalid ") + key);
    }
    if (std::getline(input, token, ';'))
        throw std::invalid_argument(std::string(key) + " requires three values");
    return fallback;
}

// Semicolon-separated class ids; a missing key selects every class.
std::vector<bool> classes(
    const std::unordered_map<std::string, std::string>& values, const char* key, int count
) {
    std::vector<bool> out(count, true);
    auto it = values.find(key);
    if (it == values.end())
        return out;
    out.assign(count, false);
    std::istringstream input(it->second);
    std::string token;
    while (std::getline(input, token, ';')) {
        if (token.empty())
            continue;
        std::size_t end = 0;
        int id = std::stoi(token, &end);
        if (end != token.size() || id < 0 || id >= count)
            throw std::invalid_argument(std::string("invalid class id in ") + key);
        out[id] = true;
    }
    return out;
}

std::vector<Polygon> polygons(const Json& value) {
    if (!value.is_array())
        throw std::invalid_argument("masks must be an array of polygons");
    std::vector<Polygon> out;
    for (const auto& poly : value) {
        if (!poly.is_array() || poly.size() < 3)
            throw std::invalid_argument("mask polygon requires at least three points");
        Polygon p;
        for (const auto& point : poly) {
            if (!point.is_array() || point.size() != 2)
                throw std::invalid_argument("mask point requires x,y");
            double x = point[0].get<double>(), y = point[1].get<double>();
            if (!std::isfinite(x) || !std::isfinite(y) || x < 0 || x > 1 || y < 0 || y > 1)
                throw std::invalid_argument("mask coordinates outside [0,1]");
            p.emplace_back(x, y);
        }
        out.push_back(std::move(p));
    }
    return out;
}

Json box_json(Box b) {
    return Json::array({b.x1, b.y1, b.x2, b.y2});
}

Json plan_json(const FramePlan& p) {
    Json tiles = Json::array();
    for (const auto& t : p.plan.tiles)
        tiles.push_back(
            {{"source", box_json(t.source)}, {"mosaic", Json::array({t.x, t.y, t.w, t.h})}}
        );
    return {
        {"version", p.version},
        {"component_id", p.component_id},
        {"source_id", p.source_id},
        {"frame_num", p.frame_num},
        {"pts_ns", p.pts == UINT64_MAX ? Json(nullptr) : Json(p.pts)},
        {"source_size", Json::array({p.source_w, p.source_h})},
        {"status", status_name(p.status)},
        {"tensor_slot", p.tensor_slot < 0 ? Json(nullptr) : Json(p.tensor_slot)},
        {"canvas_size", Json::array({p.plan.canvas_w, p.plan.canvas_h})},
        {"mosaic_size", Json::array({p.plan.mosaic_w, p.plan.mosaic_h})},
        {"resized_size", Json::array({p.plan.resized_w, p.plan.resized_h})},
        {"scale", Json::array({p.plan.scale_x, p.plan.scale_y})},
        {"letterbox_offset", Json::array({p.plan.pad_x, p.plan.pad_y})},
        {"placements", tiles},
        {"motion_boxes", p.motion_boxes},
        {"regions", p.regions},
        {"full_slot", p.full_slot < 0 ? Json(nullptr) : Json(p.full_slot)},
        {"full_scale", Json::array({p.full_plan.scale_x, p.full_plan.scale_y})},
        {"full_letterbox_offset", Json::array({p.full_plan.pad_x, p.full_plan.pad_y})},
        {"timings_ms",
         {{"motion", p.motion_ms}, {"packing", p.packing_ms}, {"tensor", p.tensor_ms}}}
    };
}

void preview(Context& c, const RgbaView& view, const FramePlan& p) {
    if (c.preview_dir.empty())
        return;
    c.events << plan_json(p).dump() << '\n';
    if (!c.events)
        throw std::runtime_error("cannot write preview events");
    double now =
        p.pts == UINT64_MAX
            ? std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
                  .count()
            : p.pts / 1e9;
    auto it = c.last_preview.find(p.source_id);
    if (it != c.last_preview.end() && now >= it->second && now - it->second < 1 / c.preview_fps)
        return;
    c.last_preview[p.source_id] = now;
    auto source = c.gpu->rgba(view);
    cv::Mat rgba(view.height, view.width, CV_8UC4, source.data()), original;
    cv::cvtColor(rgba, original, cv::COLOR_RGBA2BGR);
    for (const auto& tile : p.plan.tiles)
        cv::rectangle(
            original,
            cv::Rect(tile.source.x1, tile.source.y1, tile.source.width(), tile.source.height()),
            cv::Scalar(0, 255, 0),
            2
        );
    cv::Mat canvas(c.config.canvas_h, c.config.canvas_w, CV_8UC3, cv::Scalar(114, 114, 114));
    if (p.status == Status::Active) {
        auto pixels = c.gpu->canvas(view, p.plan);
        cv::Mat image(p.plan.canvas_h, p.plan.canvas_w, CV_8UC4, pixels.data());
        cv::cvtColor(image, canvas, cv::COLOR_RGBA2BGR);
    } else
        cv::putText(
            canvas,
            std::string(status_name(p.status)) + " - no tensor",
            {12, 30},
            cv::FONT_HERSHEY_SIMPLEX,
            .6,
            cv::Scalar(255, 255, 255),
            1
        );
    auto name = c.preview_dir + "/component_" + std::to_string(c.id) + "_" + c.preview_session +
                "_source_" + std::to_string(p.source_id) + "_frame_" + std::to_string(p.frame_num) +
                "_" + std::to_string(c.preview_seq++);
    if (!cv::imwrite(name + ".source.png", original) || !cv::imwrite(name + ".mosaic.png", canvas))
        throw std::runtime_error("cannot write preview PNG");
    std::ofstream mapping(name + ".json");
    mapping << plan_json(p).dump(2) << '\n';
    if (!mapping)
        throw std::runtime_error("cannot write placement JSON");
}

struct BufferMap {
    GstBuffer* buffer;
    GstMapInfo map = GST_MAP_INFO_INIT;

    explicit BufferMap(GstBuffer* b)
        : buffer(b) {
        if (!b || !gst_buffer_map(b, &map, GST_MAP_READ))
            throw std::runtime_error("cannot map input GstBuffer");
        if (map.size < sizeof(NvBufSurface)) {
            gst_buffer_unmap(buffer, &map);
            buffer = nullptr;
            throw std::runtime_error("input is not an NvBufSurface");
        }
    }

    ~BufferMap() {
        if (buffer)
            gst_buffer_unmap(buffer, &map);
    }
};
}

struct CustomCtx {
    std::shared_ptr<Context> value;
};

extern "C" CustomCtx* initLib(CustomInitParams params) {
    try {
        auto c = std::make_shared<Context>();
        c->id = params.unique_id;
        const auto& shape = params.tensor_params.network_input_shape;
        if (params.tensor_params.network_input_order != NvDsPreProcessNetworkInputOrder_kNCHW ||
            shape.size() != 4 || shape[0] < 1 || shape[1] != 3 || shape[2] < 1 || shape[3] < 1)
            throw std::invalid_argument(
                "requires a positive NCHW tensor shape with three channels"
            );
        if (params.tensor_params.data_type != NvDsDataType_FP32 &&
            params.tensor_params.data_type != NvDsDataType_FP16)
            throw std::invalid_argument("tensor type must be FP32 or FP16");
        if (params.tensor_params.network_color_format != NvDsPreProcessFormat_RGB &&
            params.tensor_params.network_color_format != NvDsPreProcessFormat_BGR)
            throw std::invalid_argument("tensor color format must be RGB or BGR");
        c->capacity = shape[0];
        c->config.canvas_h = shape[2];
        c->config.canvas_w = shape[3];
        c->fp16 = params.tensor_params.data_type == NvDsDataType_FP16;
        c->bgr = params.tensor_params.network_color_format == NvDsPreProcessFormat_BGR;
        const auto& u = params.user_configs;
        const std::unordered_set<std::string> allowed{
            "gpu-id",
            "motion-height",
            "motion-threshold",
            "contour-area",
            "frame-alpha",
            "lightning-threshold",
            "contrast-history",
            "improve-contrast",
            "stale-frames",
            "crop-padding",
            "gap",
            "max-zoom",
            "merge-growth",
            "channel-scale",
            "channel-offset",
            "mask-polygons",
            "source-masks",
            "preview-dir",
            "preview-fps",
            "full-frame-interval",
            "full-frame-stagger",
            "region-max-misses",
            "region-padding",
            "region-match-iou",
            "region-score",
            "seed-score",
            "max-regions",
            "infer-unique-id",
            "num-classes",
            "det-score",
            "dedupe-iou",
            "mosaic-classes",
            "full-frame-classes",
            "labels-file",
            "parser-lib",
            "parser-func"
        };
        for (const auto& item : u)
            if (!allowed.count(item.first))
                throw std::invalid_argument("unknown user config: " + item.first);
        c->device = integer(u, "gpu-id", 0);
        auto& m = c->config.motion;
        m.height = integer(u, "motion-height", 100);
        m.threshold = integer(u, "motion-threshold", 30);
        m.contour_area = integer(u, "contour-area", 10);
        m.history = integer(u, "contrast-history", 50);
        m.stale_frames = integer(u, "stale-frames", 0);
        m.alpha = number(u, "frame-alpha", .01);
        m.lightning_threshold = number(u, "lightning-threshold", .8);
        int contrast = integer(u, "improve-contrast", 1);
        if (contrast != 0 && contrast != 1)
            throw std::invalid_argument("improve-contrast must be 0 or 1");
        m.improve_contrast = contrast;
        c->config.padding = number(u, "crop-padding", .15);
        c->config.gap = integer(u, "gap", 4);
        c->config.max_zoom = number(u, "max-zoom", 2);
        c->config.merge_growth = number(u, "merge-growth", 1.3);
        c->scale = triple(u, "channel-scale", c->scale);
        c->offset = triple(u, "channel-offset", c->offset);
        if (auto it = u.find("mask-polygons"); it != u.end())
            m.masks = polygons(Json::parse(it->second));
        if (auto it = u.find("source-masks"); it != u.end()) {
            auto masks = Json::parse(it->second);
            if (!masks.is_object())
                throw std::invalid_argument("source-masks must be a source-id keyed object");
            for (auto mask = masks.begin(); mask != masks.end(); ++mask) {
                std::size_t end = 0;
                auto source = std::stoul(mask.key(), &end);
                if (end != mask.key().size() || source > UINT32_MAX)
                    throw std::invalid_argument("invalid mask source ID");
                c->config.source_masks[unsigned(source)] = polygons(mask.value());
            }
        }
        c->config.full_interval = integer(u, "full-frame-interval", 0);
        int stagger = integer(u, "full-frame-stagger", 1);
        if (stagger != 0 && stagger != 1)
            throw std::invalid_argument("full-frame-stagger must be 0 or 1");
        c->config.full_stagger = stagger;
        auto& r = c->config.regions;
        r.max_misses = integer(u, "region-max-misses", r.max_misses);
        r.padding = number(u, "region-padding", r.padding);
        r.match_iou = number(u, "region-match-iou", r.match_iou);
        r.region_score = number(u, "region-score", r.region_score);
        r.seed_score = number(u, "seed-score", r.seed_score);
        r.max_regions = integer(u, "max-regions", r.max_regions);
        auto& d = c->decoder;
        d.infer_id = unsigned(integer(u, "infer-unique-id", int(d.infer_id)));
        d.num_classes = integer(u, "num-classes", d.num_classes);
        if (!d.infer_id || d.num_classes < 1)
            throw std::invalid_argument("infer-unique-id and num-classes must be positive");
        d.det_score = float(number(u, "det-score", d.det_score));
        d.dedupe_iou = float(number(u, "dedupe-iou", d.dedupe_iou));
        d.mosaic_classes = classes(u, "mosaic-classes", d.num_classes);
        d.full_classes = classes(u, "full-frame-classes", d.num_classes);
        if (auto it = u.find("labels-file"); it != u.end() && !it->second.empty()) {
            std::ifstream labels(it->second);
            if (!labels)
                throw std::invalid_argument("cannot read labels-file " + it->second);
            for (std::string line; std::getline(labels, line);)
                d.labels.push_back(line);
        }
        if (auto it = u.find("parser-lib"); it != u.end() && !it->second.empty()) {
            d.parser_lib = dlopen(it->second.c_str(), RTLD_NOW);
            if (!d.parser_lib)
                throw std::invalid_argument(std::string("cannot load parser-lib: ") + dlerror());
            auto name = u.count("parser-func") ? u.at("parser-func") : std::string("NvDsInferYoloAttr");
            d.parse = dlsym(d.parser_lib, name.c_str());
            if (!d.parse)
                throw std::invalid_argument("parser-func " + name + " not found in parser-lib");
        }
        c->processor = std::make_unique<Processor>(c->config);
        cuda_check(cudaSetDevice(c->device), "select GPU");
        c->gpu = std::make_unique<Gpu>();
        if (auto it = u.find("preview-dir"); it != u.end())
            c->preview_dir = it->second;
        c->preview_fps = number(u, "preview-fps", 1);
        if (c->preview_fps <= 0)
            throw std::invalid_argument("preview-fps must be positive");
        if (!c->preview_dir.empty()) {
            c->preview_session =
                std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
            std::filesystem::create_directories(c->preview_dir);
            c->events.open(
                c->preview_dir + "/events_" + std::to_string(c->id) + ".jsonl", std::ios::app
            );
            if (!c->events)
                throw std::runtime_error("cannot open preview event log");
        }
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (!registry[c->id].expired())
            throw std::invalid_argument("duplicate preprocessing component ID");
        registry[c->id] = c;
        return new CustomCtx{std::move(c)};
    } catch (const std::exception& e) {
        std::cerr << "vms motion init: " << e.what() << '\n';
        return nullptr;
    }
}

extern "C" void deInitLib(CustomCtx* ctx) {
    if (!ctx)
        return;
    {
        std::lock_guard<std::mutex> lock(registry_mutex);
        registry.erase(ctx->value->id);
    }
    delete ctx;
}

std::shared_ptr<Context> vms_motion::find_context(unsigned component) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it = registry.find(component);
    return it == registry.end() ? nullptr : it->second.lock();
}

extern "C" bool vms_motion_reset_source(unsigned component, unsigned source) {
    auto c = find_context(component);
    if (!c)
        return false;
    std::lock_guard<std::mutex> lock(c->mutex);
    c->processor->reset_source(source);
    c->last_preview.erase(source);
    return true;
}

extern "C" NvDsPreProcessStatus CustomTensorPreparation(
    CustomCtx* ctx,
    NvDsPreProcessBatch* batch,
    NvDsPreProcessCustomBuf*& buf,
    CustomTensorParams& tensor,
    NvDsPreProcessAcquirer* acquirer
) {
    buf = nullptr;
    if (!ctx || !batch || !acquirer)
        return NVDSPREPROCESS_INVALID_PARAMS;
    auto c = ctx->value;
    std::lock_guard<std::mutex> lock(c->mutex);
    try {
        cuda_check(cudaSetDevice(c->device), "select GPU");
        tensor.seq_params.roi_vector.clear();
        BufferMap mapped(batch->inbuf);
        auto* surface = reinterpret_cast<NvBufSurface*>(mapped.map.data);
        if (int(surface->gpuId) != c->device)
            throw std::invalid_argument("surface GPU differs from user-configs gpu-id");
        if (batch->units.size() > std::size_t(c->capacity))
            throw std::invalid_argument("batch exceeds configured tensor capacity");

        std::vector<FramePlan> plans;
        plans.reserve(batch->units.size());
        std::unordered_set<NvDsFrameMeta*> frames;
        std::size_t sample_bytes =
            std::size_t(3) * c->config.canvas_w * c->config.canvas_h * (c->fp16 ? 2 : 4);
        unsigned slot = 0;
        for (const auto& unit : batch->units) {
            if (!unit.frame_meta || unit.obj_meta || !frames.insert(unit.frame_meta).second)
                throw std::invalid_argument(
                    "configure one full-frame unit per frame, with process-on-frame=1"
                );
            SurfaceView input(surface, unit.frame_meta->batch_id);
            const auto& view = input.get();
            int sh = c->config.motion.height,
                sw = std::max(1, int(static_cast<long long>(sh) * view.width / view.height));
            auto gray_started = std::chrono::steady_clock::now();
            auto gray = c->gpu->motion_gray(view, sw, sh);
            double gray_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - gray_started
            ).count();
            auto p = c->processor->process(
                unit.frame_meta->source_id,
                unit.frame_meta->frame_num,
                unit.frame_meta->buf_pts,
                view.width,
                view.height,
                cv::Mat(sh, sw, CV_8UC1, gray.data())
            );
            p.component_id = c->id;
            p.motion_ms += gray_ms;
            // One tensor slot per plan; nvinfer returns slot j's outputs on roi_vector[j], so a frame may own two.
            auto write = [&](const Plan& plan) {
                if (slot >= unsigned(c->capacity))
                    throw std::invalid_argument(
                        "mosaic and full-frame slots exceed the network-input-shape batch"
                    );
                if (!buf) {
                    buf = acquirer->acquire();
                    if (!buf || !buf->memory_ptr)
                        throw std::runtime_error("tensor pool exhausted");
                }
                c->gpu->tensor(
                    view,
                    plan,
                    static_cast<unsigned char*>(buf->memory_ptr) + slot * sample_bytes,
                    c->fp16,
                    c->bgr,
                    c->scale,
                    c->offset
                );
                tensor.seq_params.roi_vector.push_back(unit.roi_meta);
                return int(slot++);
            };
            auto started = std::chrono::steady_clock::now();
            if (p.status == Status::Active)
                p.tensor_slot = write(p.plan);
            if (!p.full_plan.tiles.empty())
                p.full_slot = write(p.full_plan);
            p.tensor_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started
            )
                              .count();
            preview(*c, view, p);
            plans.push_back(std::move(p));
        }
        for (std::size_t i = 0; i < plans.size(); ++i)
            attach_plan(batch->batch_meta, batch->units[i].frame_meta, plans[i]);
        tensor.params.network_input_shape = {int(slot), 3, c->config.canvas_h, c->config.canvas_w};
        tensor.params.buffer_size = slot * sample_bytes;
        return slot ? NVDSPREPROCESS_SUCCESS : NVDSPREPROCESS_TENSOR_NOT_READY;
    } catch (const std::exception& e) {
        // No C++ exception may cross the SDK boundary; release a pool buffer on every failure path.
        try {
            c->gpu->synchronize();
        } catch (...) {
        }
        if (buf)
            acquirer->release(buf);
        buf = nullptr;
        tensor.seq_params.roi_vector.clear();
        std::cerr << "vms motion preprocess: " << e.what() << '\n';
        // The stock plugin can otherwise turn a custom-library error into a null tensor and
        // continue.
        return NVDSPREPROCESS_CUSTOM_TENSOR_FAILED;
    }
}
