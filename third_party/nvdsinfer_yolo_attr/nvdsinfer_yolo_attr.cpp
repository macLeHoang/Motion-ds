// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
//
// NvDsInferYoloAttr: DeepStream nvinfer instance-mask parser for ultralytics/trt_nms.py engines whose heads have
// keypoints, attributes or children (EfficientNMSX_TRT). Structure follows levipereira/nvdsinfer_yolo.
//
// DeepStream's detection struct has no room for these outputs, so each detection carries them as a float payload in
// NvDsInferInstanceMaskInfo::mask (mask_width = payload size, mask_height = 1). yolo_attr_probe.py turns the payload
// into DeepStream metadata. Payload per detection, in this order, for the outputs the engine has:
//   det_kpts      (nkpt, ndim)  x, y relative to the reported box ((x - left) / width), then the other dims raw
//   det_attrs     (na)          attribute probabilities, raw
//   det_children  (nb, 5 + ka)  presence, parent-relative tx, ty, tw, th, attribute probabilities, raw
// Box-relative values follow nvinfer's scaling of the box to frame coordinates, so the probe needs no letterbox maths.

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

namespace {

const NvDsInferLayerInfo* findLayer(std::vector<NvDsInferLayerInfo> const& layers, const char* name) {
    for (auto const& layer : layers) {
        if (layer.layerName && std::strcmp(layer.layerName, name) == 0) return &layer;
    }
    return nullptr;
}

// Dims exclude the batch axis; maxDet > 0 also requires the first dim to be the max_det rows of the other outputs
bool checkLayer(const NvDsInferLayerInfo* layer, const char* name, NvDsInferDataType type, unsigned int rank,
                unsigned int maxDet) {
    if (!layer) {
        std::cerr << "NvDsInferYoloAttr: missing output '" << name << "'" << std::endl;
        return false;
    }
    if (layer->dataType != type || layer->inferDims.numDims != rank || (maxDet && layer->inferDims.d[0] != maxDet)) {
        std::cerr << "NvDsInferYoloAttr: output '" << name << "' must be " << (type == FLOAT ? "FLOAT" : "INT32")
                  << " with " << rank << " dims" << (maxDet ? " starting with max_det" : "")
                  << " (build the engine with FP32 inputs and outputs)" << std::endl;
        return false;
    }
    return true;
}

}  // namespace

extern "C" bool NvDsInferYoloAttr(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
                                  NvDsInferNetworkInfo const& networkInfo,
                                  NvDsInferParseDetectionParams const& detectionParams,
                                  std::vector<NvDsInferInstanceMaskInfo>& objectList) {
    const NvDsInferLayerInfo* numDets = findLayer(outputLayersInfo, "num_dets");
    const NvDsInferLayerInfo* boxes = findLayer(outputLayersInfo, "det_boxes");
    const NvDsInferLayerInfo* scores = findLayer(outputLayersInfo, "det_scores");
    const NvDsInferLayerInfo* classes = findLayer(outputLayersInfo, "det_classes");
    if (!checkLayer(scores, "det_scores", FLOAT, 1, 0)) return false;
    const unsigned int maxDet = scores->inferDims.d[0];
    if (!checkLayer(numDets, "num_dets", INT32, 1, 0) || !checkLayer(boxes, "det_boxes", FLOAT, 2, maxDet) ||
        !checkLayer(classes, "det_classes", INT32, 1, maxDet)) {
        return false;
    }
    if (findLayer(outputLayersInfo, "det_masks")) {
        std::cerr << "NvDsInferYoloAttr: found det_masks; parse segment engines with NvDsInferYoloMask" << std::endl;
        return false;
    }

    // Extra outputs in trt_nms.py order; each adds numElements / maxDet floats per detection to the payload
    struct Extra {
        const char* name;
        unsigned int rank;
        const float* data;
        unsigned int size;
    } extras[] = {{"det_kpts", 3, nullptr, 0}, {"det_attrs", 2, nullptr, 0}, {"det_children", 3, nullptr, 0}};
    unsigned int payload = 0, kptDims = 0;
    for (auto& e : extras) {
        const NvDsInferLayerInfo* layer = findLayer(outputLayersInfo, e.name);
        if (!layer) continue;
        if (!checkLayer(layer, e.name, FLOAT, e.rank, maxDet)) return false;
        e.data = static_cast<const float*>(layer->buffer);
        e.size = layer->inferDims.numElements / maxDet;
        payload += e.size;
        if (&e == &extras[0]) kptDims = layer->inferDims.d[2];
    }
    if (!payload) {
        std::cerr << "NvDsInferYoloAttr: no det_kpts, det_attrs or det_children; use NvDsInferYoloNMS" << std::endl;
        return false;
    }

    const int n = std::min(static_cast<const int*>(numDets->buffer)[0], static_cast<int>(maxDet));
    const float* box = static_cast<const float*>(boxes->buffer);
    const float* score = static_cast<const float*>(scores->buffer);
    const int* cls = static_cast<const int*>(classes->buffer);
    const float w = networkInfo.width, h = networkInfo.height;
    static bool warned = false;
    for (int i = 0; i < n; ++i) {
        const int c = cls[i];
        if (c < 0 || static_cast<unsigned int>(c) >= detectionParams.numClassesConfigured) {
            if (!warned) {
                std::cerr << "NvDsInferYoloAttr: class " << c << " >= num-detected-classes "
                          << detectionParams.numClassesConfigured << ", skipping such detections" << std::endl;
                warned = true;
            }
            continue;
        }
        if (score[i] < detectionParams.perClassPreclusterThreshold[c]) continue;
        const float x1 = std::max(0.f, std::min(box[4 * i], w)), y1 = std::max(0.f, std::min(box[4 * i + 1], h));
        const float x2 = std::max(0.f, std::min(box[4 * i + 2], w)), y2 = std::max(0.f, std::min(box[4 * i + 3], h));
        if (x2 <= x1 || y2 <= y1) continue;

        NvDsInferInstanceMaskInfo obj{};
        obj.classId = c;
        obj.detectionConfidence = score[i];
        obj.left = x1;
        obj.top = y1;
        obj.width = x2 - x1;
        obj.height = y2 - y1;
        obj.mask = new float[payload];  // released by nvinfer with the object meta
        obj.mask_width = payload;
        obj.mask_height = 1;
        obj.mask_size = payload * sizeof(float);
        float* p = obj.mask;
        for (auto const& e : extras) {
            if (!e.data) continue;
            std::memcpy(p, e.data + static_cast<size_t>(i) * e.size, e.size * sizeof(float));
            if (&e == &extras[0]) {  // keypoint x, y relative to the reported box
                for (unsigned int k = 0; k + 1 < e.size; k += kptDims) {
                    p[k] = (p[k] - x1) / obj.width;
                    p[k + 1] = (p[k + 1] - y1) / obj.height;
                }
            }
            p += e.size;
        }
        objectList.push_back(obj);
    }
    return true;
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferYoloAttr);
