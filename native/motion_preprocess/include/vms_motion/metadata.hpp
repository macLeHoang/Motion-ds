#pragma once

#include "processor.hpp"

#include <gst/gst.h>
#include <nvdsmeta.h>

namespace vms_motion {
NvDsMetaType placement_meta_type();
void attach_plan(NvDsBatchMeta*, NvDsFrameMeta*, const FramePlan&);
const FramePlan* frame_plan(const NvDsFrameMeta*, unsigned component_id);
}

extern "C" {
// Thread-safe. Call on source removal, stream reset, or reconnect. False means no live context.
bool vms_motion_reset_source(unsigned component_id, unsigned source_id);
// Attach to nvdspreprocess's src pad before a tensor-input nvinfer.
// The stock plugin may attach an NvDsPreProcessBatchMeta with tensor_meta == nullptr.
GstPadProbeReturn vms_motion_strip_empty_tensors(GstPad*, GstPadProbeInfo*, gpointer);
}
