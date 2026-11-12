#include "vms_motion/metadata.hpp"

#include <stdexcept>

#include <gstnvdsmeta.h>
#include <nvdspreprocess_meta.h>

namespace vms_motion {
NvDsMetaType placement_meta_type() {
    static gchar name[] = "VMS.MOTION_MOSAIC.PLACEMENTS.V1";
    static auto type = nvds_get_user_meta_type(name);
    return type;
}

static gpointer copy_plan(gpointer data, gpointer) {
    auto* user = static_cast<NvDsUserMeta*>(data);
    try {
        return new FramePlan(*static_cast<FramePlan*>(user->user_meta_data));
    } catch (...) {
        return nullptr;
    }
}

static void release_plan(gpointer data, gpointer) {
    auto* user = static_cast<NvDsUserMeta*>(data);
    delete static_cast<FramePlan*>(user->user_meta_data);
    user->user_meta_data = nullptr;
}

void attach_plan(NvDsBatchMeta* batch, NvDsFrameMeta* frame, const FramePlan& plan) {
    nvds_acquire_meta_lock(batch);
    auto* user = nvds_acquire_user_meta_from_pool(batch);
    if (!user) {
        nvds_release_meta_lock(batch);
        throw std::runtime_error("placement metadata pool exhausted");
    }
    try {
        user->user_meta_data = new FramePlan(plan);
    } catch (...) {
        nvds_release_meta_lock(batch);
        throw;
    }
    user->base_meta.meta_type = placement_meta_type();
    user->base_meta.copy_func = copy_plan;
    user->base_meta.release_func = release_plan;
    nvds_add_user_meta_to_frame(frame, user);
    nvds_release_meta_lock(batch);
}

const FramePlan* frame_plan(const NvDsFrameMeta* frame, unsigned component) {
    for (auto* l = frame->frame_user_meta_list; l; l = l->next) {
        auto* u = static_cast<NvDsUserMeta*>(l->data);
        if (u->base_meta.meta_type == placement_meta_type()) {
            auto* p = static_cast<FramePlan*>(u->user_meta_data);
            if (p && p->version == 1 && p->component_id == component)
                return p;
        }
    }
    return nullptr;
}
}

extern "C" GstPadProbeReturn
vms_motion_strip_empty_tensors(GstPad* pad, GstPadProbeInfo* info, gpointer component) {
    if (!(GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER))
        return GST_PAD_PROBE_OK;
    auto* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    auto* batch = buffer ? gst_buffer_get_nvds_batch_meta(buffer) : nullptr;
    if (!batch)
        return GST_PAD_PROBE_OK;
    // Passing the configured component ID also makes missing placement metadata fatal.
    // This surfaces custom-library failures on SDK versions that forward a null tensor.
    if (component) {
        for (auto* l = batch->frame_meta_list; l; l = l->next) {
            auto* frame = static_cast<NvDsFrameMeta*>(l->data);
            if (!vms_motion::frame_plan(frame, GPOINTER_TO_UINT(component))) {
                auto* owner = gst_pad_get_parent_element(pad);
                if (owner) {
                    GST_ELEMENT_ERROR(
                        owner,
                        STREAM,
                        FAILED,
                        ("Motion placement metadata is missing"),
                        ("Check the custom-library error log and source-group configuration")
                    );
                    gst_object_unref(owner);
                }
                return GST_PAD_PROBE_DROP;
            }
        }
    }
    nvds_acquire_meta_lock(batch);
    for (auto* l = batch->batch_user_meta_list; l;) {
        auto* next = l->next;
        auto* u = static_cast<NvDsUserMeta*>(l->data);
        if (u->base_meta.meta_type == NVDS_PREPROCESS_BATCH_META) {
            auto* pre = static_cast<GstNvDsPreProcessBatchMeta*>(u->user_meta_data);
            if (pre && !pre->tensor_meta)
                nvds_remove_user_meta_from_batch(batch, u);
        }
        l = next;
    }
    nvds_release_meta_lock(batch);
    return GST_PAD_PROBE_OK;
}
