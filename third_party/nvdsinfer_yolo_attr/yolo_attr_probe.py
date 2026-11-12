# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Turn NvDsInferYoloAttr payloads into DeepStream metadata: attribute labels, child objects and keypoint circles.

`decode` is plain numpy; `attach` and `probe` import pyds and GStreamer only when called.

Examples:
    >>> meta = load_meta("ialy_v9_clean_erd_m_last-trt.onnx")
    >>> nvinfer.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, probe, meta)
"""

from __future__ import annotations

import ast
import math

import numpy as np

UNTRACKED_OBJECT_ID = 0xFFFFFFFFFFFFFFFF


def _literal(v: str):
    """Parse a metadata value written with str(), keeping plain text as it is."""
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def load_meta(onnx_path: str) -> dict:
    """Read names, masks and the payload layout from a trt_nms `-trt.onnx` export's metadata.

    Args:
        onnx_path (str): The `-trt.onnx` file the engine was built from.

    Returns:
        (dict): The export metadata, plus `layout` [(group, shape)] in payload order and `payload_size` in floats.
    """
    import onnx

    meta = {p.key: _literal(p.value) for p in onnx.load(str(onnx_path), load_external_data=False).metadata_props}
    layout = []
    if "kpt_shape" in meta:
        layout.append(("kpts", tuple(meta["kpt_shape"])))
    if "attributes" in meta:
        layout.append(("attrs", (len(meta["attributes"]),)))
        meta["attr_mask"] = np.asarray(meta["attr_mask"], np.float32)  # (nc, na)
    if "child_names" in meta:
        layout.append(("children", (len(meta["child_names"]), 5 + len(meta["child_attr_names"]))))
        meta["child_mask"] = np.asarray(meta["child_mask"], np.float32)  # (nc, nb)
        meta["child_attr_mask"] = np.asarray(meta["child_attr_mask"], np.float32)  # (nb, ka)
    meta["layout"], meta["payload_size"] = layout, sum(math.prod(s) for _, s in layout)
    return meta


def decode(payload, rect: tuple, class_id: int, meta: dict) -> dict:
    """Decode one object's payload into frame coordinates, as `get_attributes` in Ultralytics predict does.

    Args:
        payload (np.ndarray): The object's NvDsInferYoloAttr payload, `obj_meta.mask_params.get_mask_array()`.
        rect (tuple): Object box (left, top, width, height) in frame pixels, `obj_meta.rect_params`.
        class_id (int): Object class index.
        meta (dict): `load_meta` result.

    Returns:
        (dict): For the groups the model has: `kpts` (nkpt, ndim), `attrs` (na,), `child_boxes` (nb, 4) xyxy,
            `child_conf` (nb,) and `child_attrs` (nb, ka). Attributes and children a class does not have are 0.
    """
    p = np.asarray(payload, np.float32).ravel()
    assert p.size == meta["payload_size"], f"payload has {p.size} floats, metadata layout {meta['layout']} needs more"
    left, top, w, h = rect
    out, i = {}, 0
    for group, shape in meta["layout"]:
        v = p[i : i + math.prod(shape)].reshape(shape)
        i += v.size
        if group == "kpts":  # x, y are relative to the box
            v = v.copy()
            v[:, 0], v[:, 1] = left + v[:, 0] * w, top + v[:, 1] * h
            out["kpts"] = v
        elif group == "attrs":
            out["attrs"] = v * meta["attr_mask"][class_id]
        else:  # presence, parent-relative tx, ty, tw, th, attributes
            cx, cy = left + w / 2 + v[:, 1] * w, top + h / 2 + v[:, 2] * h
            cw, ch = w * np.exp(np.minimum(v[:, 3], 4.0)), h * np.exp(np.minimum(v[:, 4], 4.0))
            out["child_boxes"] = np.stack([cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2], 1)
            out["child_conf"] = v[:, 0] * meta["child_mask"][class_id]
            out["child_attrs"] = v[:, 5:] * meta["child_attr_mask"]
    return out


def _add_labels(pyds, batch_meta, obj, probs, names: list, thr: float) -> None:
    """Attach the attributes above `thr` to `obj` as one classifier meta with a label per attribute."""
    keep = np.flatnonzero(probs > thr)
    if not keep.size:
        return
    cls_meta = pyds.nvds_acquire_classifier_meta_from_pool(batch_meta)
    cls_meta.unique_component_id = obj.unique_component_id
    for k in keep:
        label = pyds.nvds_acquire_label_info_meta_from_pool(batch_meta)
        label.result_label = names[k]
        label.result_class_id = int(k)
        label.result_prob = float(probs[k])
        pyds.nvds_add_label_info_meta_to_classifier(cls_meta, label)
    pyds.nvds_add_classifier_meta_to_object(obj, cls_meta)


def attach(
    batch_meta,
    meta: dict,
    attr_thr: float = 0.5,
    child_thr: float = 0.5,
    kpt_thr: float = 0.5,
    kpt_classes: set | None = None,
) -> None:
    """Add attribute labels, child objects and keypoint circles for every NvDsInferYoloAttr object in a batch.

    Attributes become a classifier meta on the object. Each present child becomes an object with `parent` set, class
    `nc + slot` and its own attribute labels. Keypoints with visibility above `kpt_thr` are drawn as display circles.

    Args:
        batch_meta (pyds.NvDsBatchMeta): Batch metadata of the buffer after nvinfer (or nvtracker).
        meta (dict): `load_meta` result.
        attr_thr (float): Minimum probability for an attribute label.
        child_thr (float): Minimum presence for a child object.
        kpt_thr (float): Minimum visibility for a drawn keypoint, when keypoints have a visibility value.
        kpt_classes (set | None): Class ids whose keypoints are drawn, e.g. the plate class. Classes trained without
            keypoint labels still output points, so None (every class) draws those too.
    """
    import pyds

    nc, size = len(meta["names"]), meta["payload_size"] * 4  # mask_params.size is in bytes
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame = pyds.NvDsFrameMeta.cast(l_frame.data)
        objs, l_obj = [], frame.obj_meta_list
        while l_obj is not None:  # collect first: children are added to the same list
            objs.append(pyds.NvDsObjectMeta.cast(l_obj.data))
            l_obj = l_obj.next
        points = []
        for obj in objs:
            if obj.mask_params.size != size:  # tracker-only object, a child, or another model's object
                continue
            r = obj.rect_params
            d = decode(obj.mask_params.get_mask_array(), (r.left, r.top, r.width, r.height), obj.class_id, meta)
            if "attrs" in d:
                _add_labels(pyds, batch_meta, obj, d["attrs"], meta["attributes"], attr_thr)
            for slot in np.flatnonzero(d.get("child_conf", np.zeros(0)) > child_thr):
                child = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
                child.unique_component_id = obj.unique_component_id
                child.class_id = nc + int(slot)
                child.object_id = UNTRACKED_OBJECT_ID
                child.confidence = float(d["child_conf"][slot])
                child.obj_label = meta["child_names"][slot]
                x1, y1, x2, y2 = d["child_boxes"][slot].tolist()
                cr = child.rect_params
                cr.left, cr.top, cr.width, cr.height = x1, y1, x2 - x1, y2 - y1
                cr.border_width = 2
                cr.border_color.set(0.0, 1.0, 0.0, 1.0)
                _add_labels(pyds, batch_meta, child, d["child_attrs"][slot], meta["child_attr_names"], attr_thr)
                pyds.nvds_add_obj_meta_to_frame(frame, child, obj)
            if "kpts" in d and (kpt_classes is None or obj.class_id in kpt_classes):
                k = d["kpts"]
                points += k[k[:, 2] > kpt_thr, :2].tolist() if k.shape[1] > 2 else k.tolist()
        display = None
        for i, (x, y) in enumerate(points):
            if i % 16 == 0:  # MAX_ELEMENTS_IN_DISPLAY_META circles per display meta
                display = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
                pyds.nvds_add_display_meta_to_frame(frame, display)
            circle = display.circle_params[display.num_circles]
            circle.xc, circle.yc, circle.radius = max(int(x), 0), max(int(y), 0), 3
            circle.circle_color.set(1.0, 0.0, 1.0, 1.0)
            display.num_circles += 1
        l_frame = l_frame.next


def probe(pad, info, meta: dict, **kwargs):
    """GStreamer buffer pad probe calling `attach(..., **kwargs)`; add it on nvinfer's (or nvtracker's) src pad.

    Use `functools.partial(probe, kpt_classes={plate_id})` to pass `attach` options.
    """
    import gi

    gi.require_version("Gst", "1.0")
    import pyds
    from gi.repository import Gst

    buf = info.get_buffer()
    if buf is not None:
        attach(pyds.gst_buffer_get_nvds_batch_meta(hash(buf)), meta, **kwargs)
    return Gst.PadProbeReturn.OK
