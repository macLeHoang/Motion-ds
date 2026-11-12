// Test-only parser: permit an identity network to pass through nvinfer's detector interface.
#include <nvdsinfer_custom_impl.h>

extern "C" bool VmsIdentityParse(
    std::vector<NvDsInferLayerInfo> const&,
    NvDsInferNetworkInfo const&,
    NvDsInferParseDetectionParams const&,
    std::vector<NvDsInferObjectDetectionInfo>& boxes
) {
    boxes.clear();
    return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(VmsIdentityParse);
