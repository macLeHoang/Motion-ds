// Decode probe for the nvinfer that reads this library's tensors (input-tensor-from-meta=1, network-type=100,
// output-tensor-meta=1). For every tensor slot it runs the configured instance-mask parser (NvDsInferYoloAttr) on the
// slot's outputs, maps the boxes from canvas to frame pixels through the slot's plan, keeps the configured classes,
// removes cross-crop duplicates, attaches the objects like nvinfer would (payload in mask_params), and feeds the
// person boxes back to the source's region tracker.

#include "context.hpp"
#include "vms_motion/metadata.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>

#include <gstnvdsinfer.h>
#include <gstnvdsmeta.h>
#include <nvdsinfer_custom_impl.h>
#include <nvdspreprocess_meta.h>

using namespace vms_motion;

namespace {
enum Origin : gint64 { Mosaic = 1, FullFrame = 2 };  // stored in misc_obj_info[0]

struct Det {
    NvDsInferInstanceMaskInfo obj{};
    std::unique_ptr<float[]> payload;  // the parser's new[] allocation
    float box[4]{};                    // frame pixels x1, y1, x2, y2
    Origin origin = Mosaic;
};

float area(const float* b) {
    return std::max(0.f, b[2] - b[0]) * std::max(0.f, b[3] - b[1]);
}

// Same label and IoU >= iou_min, or one box 80 % inside the other (dls/core/regions.py dedupe_detections).
bool duplicate(const Det& a, const Det& b, float iou_min) {
    if (a.obj.classId != b.obj.classId)
        return false;
    float iw = std::max(0.f, std::min(a.box[2], b.box[2]) - std::max(a.box[0], b.box[0])),
          ih = std::max(0.f, std::min(a.box[3], b.box[3]) - std::max(a.box[1], b.box[1])), inter = iw * ih,
          aa = area(a.box), ab = area(b.box);
    return inter / (aa + ab - inter + 1e-9f) >= iou_min || inter >= .8f * std::min(aa, ab);
}

// Canvas box -> frame pixels through the tile holding its centre (geometry.cpp map_box, without clipping to the crop
// so the box-relative payload stays exact); clipped to the frame as nvinfer does. False when the centre is in a gap.
bool to_frame(const Plan& p, int w, int h, const NvDsInferInstanceMaskInfo& o, float* out) {
    double cx = (o.left + o.width / 2 - p.pad_x) / p.scale_x, cy = (o.top + o.height / 2 - p.pad_y) / p.scale_y;
    for (const auto& t : p.tiles) {
        if (cx < t.x || cx >= t.x + t.w || cy < t.y || cy >= t.y + t.h)
            continue;
        auto lo = map_point(p, t, o.left, o.top), hi = map_point(p, t, o.left + o.width, o.top + o.height);
        out[0] = float(std::clamp(lo[0], 0., double(w)));
        out[1] = float(std::clamp(lo[1], 0., double(h)));
        out[2] = float(std::clamp(hi[0], 0., double(w)));
        out[3] = float(std::clamp(hi[1], 0., double(h)));
        return out[2] > out[0] && out[3] > out[1];
    }
    return false;
}

NvDsInferTensorMeta* slot_outputs(NvDsRoiMeta& roi, unsigned infer_id) {
    for (auto* l = roi.roi_user_meta_list; l; l = l->next) {
        auto* u = static_cast<NvDsUserMeta*>(l->data);
        if (u->base_meta.meta_type == NVDSINFER_TENSOR_OUTPUT_META) {
            auto* t = static_cast<NvDsInferTensorMeta*>(u->user_meta_data);
            if (t && t->unique_id == infer_id)
                return t;
        }
    }
    return nullptr;
}

// Runs the parser on one slot and keeps the detections whose centre lands in a tile of `plan`.
std::vector<Det> parse_slot(const DecoderConfig& d, const NvDsInferTensorMeta& t, const Plan& plan, int w, int h,
                            Origin origin) {
    std::vector<NvDsInferLayerInfo> layers(t.output_layers_info, t.output_layers_info + t.num_output_layers);
    for (unsigned i = 0; i < t.num_output_layers; ++i)
        layers[i].buffer = t.out_buf_ptrs_host[i];
    NvDsInferNetworkInfo net{unsigned(plan.canvas_w), unsigned(plan.canvas_h), 3};
    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = unsigned(d.num_classes);
    params.perClassPreclusterThreshold.assign(d.num_classes, d.det_score);
    params.perClassPostclusterThreshold.assign(d.num_classes, 0.f);
    std::vector<NvDsInferInstanceMaskInfo> objects;
    auto parse = reinterpret_cast<NvDsInferInstanceMaskParseCustomFunc>(d.parse);
    bool ok = parse(layers, net, params, objects);
    std::vector<Det> out;
    for (auto& o : objects) {
        Det det;
        det.obj = o;
        det.payload.reset(o.mask);
        det.origin = origin;
        if (ok && to_frame(plan, w, h, o, det.box))
            out.push_back(std::move(det));
    }
    if (!ok)
        throw std::runtime_error("parser-func rejected the engine outputs");
    return out;
}

void attach(NvDsBatchMeta* batch, NvDsFrameMeta* frame, const Det& det, const DecoderConfig& d) {
    auto* obj = nvds_acquire_obj_meta_from_pool(batch);
    const auto& o = det.obj;
    obj->unique_component_id = int(d.infer_id);
    obj->class_id = int(o.classId);
    obj->confidence = o.detectionConfidence;
    obj->object_id = UNTRACKED_OBJECT_ID;
    obj->misc_obj_info[0] = det.origin;
    auto& r = obj->rect_params;
    r.left = det.box[0];
    r.top = det.box[1];
    r.width = det.box[2] - det.box[0];
    r.height = det.box[3] - det.box[1];
    r.border_width = 2;
    r.border_color = det.origin == Mosaic ? NvOSD_ColorParams{0, 1, 0, 1} : NvOSD_ColorParams{1, .5, 0, 1};
    obj->detector_bbox_info.org_bbox_coords = {r.left, r.top, r.width, r.height};
    std::string label = o.classId < d.labels.size() ? d.labels[o.classId] : std::to_string(o.classId);
    g_strlcpy(obj->obj_label, label.c_str(), MAX_LABEL_SIZE);
    auto& text = obj->text_params;
    text.display_text = g_strdup_printf("%s %.2f", label.c_str(), o.detectionConfidence);
    text.x_offset = unsigned(std::max(0.f, r.left));
    text.y_offset = unsigned(std::max(0.f, r.top - 10));
    text.font_params.font_name = const_cast<gchar*>("Serif");
    text.font_params.font_size = 11;
    text.font_params.font_color = {1, 1, 1, 1};
    text.set_bg_clr = 1;
    text.text_bg_clr = {0, 0, 0, 1};
    if (det.payload && o.mask_size) {
        auto* payload = static_cast<float*>(g_malloc(o.mask_size));  // freed with the object meta, as nvinfer's
        std::memcpy(payload, det.payload.get(), o.mask_size);
        obj->mask_params.data = payload;
        obj->mask_params.size = o.mask_size;
        obj->mask_params.width = o.mask_width;
        obj->mask_params.height = o.mask_height;
        obj->mask_params.threshold = 0;
    }
    nvds_add_obj_meta_to_frame(frame, obj, nullptr);
}

Box int_box(const float* b) {
    return {int(b[0]), int(b[1]), int(b[2]), int(b[3])};
}

struct FrameOut {
    const FramePlan* plan = nullptr;
    bool mosaic_ran = false;
    std::vector<Det> mosaic, full;
};

void decode_batch(Context& c, NvDsBatchMeta* batch) {
    const auto& d = c.decoder;
    if (!d.parse)
        throw std::invalid_argument("decode probe installed but parser-lib is not configured");
    std::map<NvDsFrameMeta*, FrameOut> frames;
    for (auto* l = batch->batch_user_meta_list; l; l = l->next) {
        auto* user = static_cast<NvDsUserMeta*>(l->data);
        if (user->base_meta.meta_type != NVDS_PREPROCESS_BATCH_META)
            continue;
        auto* pre = static_cast<GstNvDsPreProcessBatchMeta*>(user->user_meta_data);
        if (!pre || !pre->tensor_meta)
            continue;
        for (std::size_t j = 0; j < pre->roi_vector.size(); ++j) {
            auto& roi = pre->roi_vector[j];
            const auto* plan = roi.frame_meta ? frame_plan(roi.frame_meta, c.id) : nullptr;
            if (!plan || (int(j) != plan->tensor_slot && int(j) != plan->full_slot))
                continue;  // another component's tensor
            auto* outputs = slot_outputs(roi, d.infer_id);
            if (!outputs)
                throw std::runtime_error("no nvinfer output for a tensor slot (check infer-unique-id)");
            auto& f = frames[roi.frame_meta];
            f.plan = plan;
            bool mosaic = int(j) == plan->tensor_slot;
            f.mosaic_ran |= mosaic;
            auto dets = parse_slot(d, *outputs, mosaic ? plan->plan : plan->full_plan, plan->source_w,
                                   plan->source_h, mosaic ? Mosaic : FullFrame);
            for (auto& det : dets)
                (mosaic ? f.mosaic : f.full).push_back(std::move(det));
        }
    }
    auto by_score = [](const Det& a, const Det& b) { return a.obj.detectionConfidence > b.obj.detectionConfidence; };
    for (auto& [frame, f] : frames) {
        std::vector<Det> kept;
        std::vector<Detection> persons, seeds;
        std::stable_sort(f.mosaic.begin(), f.mosaic.end(), by_score);
        for (auto& det : f.mosaic) {
            bool keep = det.obj.classId < d.mosaic_classes.size() && d.mosaic_classes[det.obj.classId] &&
                        std::none_of(kept.begin(), kept.end(), [&](const Det& k) {
                            return duplicate(det, k, d.dedupe_iou);
                        });
            if (keep) {
                persons.push_back({int_box(det.box), det.obj.detectionConfidence});
                kept.push_back(std::move(det));
            }
        }
        const std::size_t from_mosaic = kept.size();
        for (auto& det : f.full) {
            auto c_id = det.obj.classId;
            bool region_class = c_id < d.mosaic_classes.size() && d.mosaic_classes[c_id];
            if (region_class)
                seeds.push_back({int_box(det.box), det.obj.detectionConfidence});
            // Full-frame boxes of the mosaic classes only where the mosaic has none (people not yet in a region).
            bool keep = (c_id < d.full_classes.size() && d.full_classes[c_id]) ||
                        (region_class && std::none_of(kept.begin(), kept.begin() + from_mosaic, [&](const Det& k) {
                             return duplicate(det, k, d.dedupe_iou);
                         }));
            if (keep)
                kept.push_back(std::move(det));
        }
        for (const auto& det : kept)
            attach(batch, frame, det, d);
        c.processor->feedback(frame->source_id, frame->frame_num, f.mosaic_ran, persons, seeds);
    }
}
}

extern "C" GstPadProbeReturn vms_motion_decode_probe(GstPad* pad, GstPadProbeInfo* info, gpointer component) {
    if (!(GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER))
        return GST_PAD_PROBE_OK;
    auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    auto* batch = buffer ? gst_buffer_get_nvds_batch_meta(buffer) : nullptr;
    auto c = batch ? find_context(GPOINTER_TO_UINT(component)) : nullptr;
    if (!c)
        return GST_PAD_PROBE_OK;
    try {
        std::lock_guard<std::mutex> lock(c->mutex);
        decode_batch(*c, batch);
    } catch (const std::exception& e) {
        std::cerr << "vms motion decode: " << e.what() << '\n';
        if (auto* owner = pad ? gst_pad_get_parent_element(pad) : nullptr) {
            GST_ELEMENT_ERROR(owner, STREAM, FAILED, ("Motion mosaic decode failed"), ("%s", e.what()));
            gst_object_unref(owner);
        }
        return GST_PAD_PROBE_DROP;
    }
    return GST_PAD_PROBE_OK;
}

extern "C" bool vms_motion_install_probes(GstPad* preprocess_src, GstPad* infer_src, unsigned component) {
    if (!preprocess_src || !infer_src || !component)
        return false;
    gst_pad_add_probe(preprocess_src, GST_PAD_PROBE_TYPE_BUFFER, vms_motion_strip_empty_tensors,
                      GUINT_TO_POINTER(component), nullptr);
    gst_pad_add_probe(infer_src, GST_PAD_PROBE_TYPE_BUFFER, vms_motion_decode_probe, GUINT_TO_POINTER(component),
                      nullptr);
    return true;
}
