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
// Attach to the tensor-input nvinfer's src pad (output-tensor-meta=1, network-type=100), probe data =
// GUINT_TO_POINTER(component_id). Parses every mosaic / full-frame slot with the configured parser-lib, adds frame-pixel
// objects (payload in mask_params, origin in misc_obj_info[0]: 1 mosaic, 2 full frame) and feeds regions back.
GstPadProbeReturn vms_motion_decode_probe(GstPad*, GstPadProbeInfo*, gpointer);
// Installs vms_motion_strip_empty_tensors and vms_motion_decode_probe; for callers without C function pointers (Python).
bool vms_motion_install_probes(GstPad* preprocess_src, GstPad* infer_src, unsigned component_id);
}
