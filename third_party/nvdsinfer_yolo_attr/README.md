# NvDsInferYoloAttr — DeepStream parser for attribute, child and pose engines

DeepStream `nvinfer` parser for engines exported with `ultralytics/trt_nms.py` (see `TRT_NMS.md`) whose head has
keypoints, attributes or children, such as `PoseChild`. These engines use EfficientNMSX_TRT and return
`num_dets, det_boxes, det_scores, det_classes` plus `det_kpts`, `det_attrs` and `det_children`.

DeepStream's detection struct has no room for the extra outputs. `NvDsInferYoloAttr` therefore hands each detection's
extras to `nvinfer` as a float payload in the instance-mask buffer, and `yolo_attr_probe.py` turns that payload into
native DeepStream metadata:

- attributes become classifier labels on the object
- children become objects with `parent` set
- keypoints become display circles

For other exports, use levi's [nvdsinfer_yolo](https://github.com/levipereira/nvdsinfer_yolo): `NvDsInferYoloNMS` for
plain detect, `NvDsInferYoloMask` for segment.

| File                         | Content                                                    |
| ---------------------------- | ---------------------------------------------------------- |
| `nvdsinfer_yolo_attr.cpp`    | `NvDsInferYoloAttr`, an instance-mask parse function       |
| `Makefile`                   | Builds `libnvds_infer_yolo_attr.so`                        |
| `config_infer_yolo_attr.txt` | Sample `nvinfer` config                                    |
| `yolo_attr_probe.py`         | `load_meta`, `decode` (numpy), `attach` and `probe` (pyds) |

---

## 1. Build

```bash
cp -R deepstream/nvdsinfer_yolo_attr /opt/nvidia/deepstream/deepstream/sources/libs/
cd /opt/nvidia/deepstream/deepstream/sources/libs/nvdsinfer_yolo_attr
CUDA_VER=12.6 make install                            # parser only
CUDA_VER=12.6 NMSX_LIB=/path/libnmsx.so make install  # also links your EfficientNMSX_TRT plugin library
```

`nvinfer` must have EfficientNMSX_TRT registered when it loads the engine. With `NMSX_LIB`, loading the parser
library loads the plugin too. Without it, start the app with `LD_PRELOAD=/path/libnmsx.so`.

---

## 2. Engine and labels

Export the ONNX as described in `TRT_NMS.md` §1.1. Then build the engine on the DeepStream machine, with a fixed
network size and a dynamic batch. As far as we know, `nvinfer` runs one network resolution, and fixing it also avoids
the child-head minimum size (`TRT_NMS.md` §4.3):

```bash
trtexec --onnx=ialy_v9_clean_erd_m_last-trt.onnx --saveEngine=ialy_v9_clean_erd_m_last-trt.engine --fp16 \
    --staticPlugins=/path/libnmsx.so \
    --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640
```

Keep the inputs and outputs FP32: `--fp16` alone does that, so don't add `--outputIOFormats`. The parser refuses
FP16 outputs.

Write `labels.txt` from the export metadata (`pip install onnx` in the DeepStream environment):

```python
from yolo_attr_probe import load_meta

meta = load_meta("ialy_v9_clean_erd_m_last-trt.onnx")
open("labels.txt", "w").write("\n".join(meta["names"][i] for i in range(len(meta["names"]))))
```

---

## 3. `nvinfer` config

Start from `config_infer_yolo_attr.txt`, and set `model-engine-file`, `batch-size` (at most the engine's maximum
batch) and `num-detected-classes`. The settings that matter:

| Key                                                       | Why                                                                   |
| --------------------------------------------------------- | --------------------------------------------------------------------- |
| `network-type=3`, `output-instance-mask=1`                | Lets the parser attach its per-object payload; nothing is a real mask |
| `parse-bbox-instance-mask-func-name=NvDsInferYoloAttr`    | The parse function                                                    |
| `cluster-mode=4`                                          | No clustering: NMS already ran in the engine                          |
| `model-color-format=0`, `net-scale-factor=1/255`          | RGB, 0–1 input, as in Ultralytics                                     |
| `maintain-aspect-ratio=1`, `symmetric-padding=1`          | Centered letterbox, as in Ultralytics                                 |
| `pre-cluster-threshold` (per class, `[class-attrs-<id>]`) | The parser applies each class's own threshold                         |

Keep `nvdsosd`'s `display-mask` off, so the payload is never drawn as a mask.

---

## 4. Probe

```python
import functools

from yolo_attr_probe import load_meta, probe

meta = load_meta("ialy_v9_clean_erd_m_last-trt.onnx")
pad = tracker.get_static_pad("src")  # or pgie.get_static_pad("src") when there is no tracker
pad.add_probe(Gst.PadProbeType.BUFFER, functools.partial(probe, kpt_classes={6}), meta)  # 6: license_plate
```

For every object carrying a payload, `attach` (called by `probe`) adds:

| Output     | DeepStream metadata                                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Attributes | One `NvDsClassifierMeta` on the object, with an `NvDsLabelInfo` per attribute above `attr_thr`                                                                |
| Children   | An `NvDsObjectMeta` per child above `child_thr`: `parent` = the object, `class_id` = `nc + slot`, label = child name, green box, and its own attribute labels |
| Keypoints  | Display-meta circles for points with visibility above `kpt_thr`, only for `kpt_classes`                                                                       |

Downstream code can read attributes from the object's classifier meta and find children through their `parent`. To
use the values directly instead, call `decode(obj.mask_params.get_mask_array(), rect, obj.class_id, meta)`. It
returns keypoints, masked attributes, child boxes, child conf and child attributes in frame pixels, with the same maths
as Ultralytics predict.

Placement:

- **After `nvtracker`** (recommended): parents get track IDs, and children are added after tracking, so they don't get
  their own.
- **After `nvinfer`:** children are tracked as independent objects.
- **Tracker-only objects** (no detection in that frame) carry no payload and are skipped.

---

## 5. Payload layout

Per detection, `mask_width` floats (`mask_height = 1`), for the outputs the engine has, in this order:

| Group      | Floats          | Encoding                                                                                 |
| ---------- | --------------- | ---------------------------------------------------------------------------------------- |
| Keypoints  | `nkpt × ndim`   | `(x - left) / width`, `(y - top) / height` against the reported box, then visibility raw |
| Attributes | `na`            | Probabilities, raw (`decode` multiplies by `attr_mask[class]`)                           |
| Children   | `nb × (5 + ka)` | Per slot: presence, parent-relative `tx, ty, tw, th`, attribute probabilities, raw       |

For the `PoseChild` checkpoints that's 12 + 43 + 105 = 160 floats. `load_meta` derives the same layout from the ONNX
metadata, and `decode` asserts that the payload size matches.

Keypoints and children are relative to the box. `nvinfer` maps boxes to the frame with a scale and an offset per axis,
so decoding against the final `rect_params` gives frame coordinates without redoing the letterbox maths.

---

## 6. Limits

- **Frame-border clipping:** if `nvinfer` clips a box at the frame edge after the parser has reported it, that
  object's keypoints and children are decoded against the smaller box and shift slightly.
- **Padding colour:** DeepStream pads the letterbox with black, while Ultralytics trained with gray (114), so scores
  differ slightly near image borders.
- **Keypoints on other classes:** classes trained without keypoint labels still output points with high visibility
  (e.g. 28 on backpacks in `zidane.jpg`). Restrict drawing with `kpt_classes`.
- **Child names** come from the checkpoint's `child_names`. If they're out of slot order, every child is mislabelled.
  `ialy_v9_clean_erd_m` stores them alphabetically: its `foot_left` slot lands at the top of person boxes.
- **Segment engines** (`det_masks`) are refused; use `NvDsInferYoloMask`.

---

## 7. Status

Verified without DeepStream, with scratch stand-ins for the SDK headers and pyds:

- **Compile:** the parser builds with `g++ -std=c++11 -Wall -Wextra` against stand-in headers with the DeepStream 6/7
  type layout. The parser code itself raises no warnings.
- **Parser + `decode` vs `YOLO.predict`:** checked on `ialy_v9_clean_erd_m`, with a real NMS standing in for the plugin
  and `bus.jpg` + `zidane.jpg` as one batch (57 detections), each frame passed as `nvinfer` does. Mapped to frame
  pixels, classes, scores, masked attributes, child presence and child attributes equal predict. Keypoints are within
  2e-4 px, and child boxes within 4e-4 px.
- **Parser options and refusals:** per-class thresholds and `num-detected-classes` filtering work. FP16 outputs,
  `det_masks`, and engines without extras are refused with a message.
- **`attach`:** run through a recording pyds stand-in, it produces exactly the expected labels, children (all with
  their parent) and keypoint circles, and skips objects without a payload.

Not yet verified: the real DeepStream headers and `nvinfer` behaviour, the real pyds bindings, and a full pipeline on
a GPU.
