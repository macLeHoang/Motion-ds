"""DeepStream motion-mosaic test app.

nvurisrcbin x N -> nvstreammux -> nvdspreprocess (motion mosaic + person regions + staggered full frame, native
library) -> one nvinfer (raw tensors) -> vms_motion_decode_probe (C++: NvDsInferYoloAttr per slot, map back, objects,
region feedback) -> nvtracker -> dump probe + yolo_attr_probe.attach -> one annotated mp4 per input (or fakesink).

Writes to --output: per input `<stem>_ds.mp4`, `<stem>_engine_dets.jsonl` (detections after the tracker),
`<stem>_person_raw.npz` (persons with decoded attributes/children, the analysis scripts' format), `plans/events_1.jsonl`
(the library's per-frame plan: mosaic/full-frame slots, regions) and `summary.json`.

    scripts/docker.sh python3 app/motion_ds_app.py --config configs/app.yaml --output runs/test \\
        --input /data/video_test/camera-1-20261111-221001-20261111-221701.mp4
"""

from __future__ import annotations

import argparse
import configparser
import ctypes
import functools
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import gi
import numpy as np
import yaml

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst  # noqa: E402

import pyds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "third_party/nvdsinfer_yolo_attr"), str(ROOT / "analysis")]
import yolo_attr_probe  # noqa: E402
from display import display_masks  # noqa: E402

ORIGIN = {1: "mosaic", 2: "full", 3: "held"}  # misc_obj_info[0], set by vms_motion_decode_probe
UNTRACKED = 0xFFFFFFFFFFFFFFFF


def load_meta(path: Path) -> dict:
    """yolo_attr_probe.load_meta's dict from a JSON copy of the export metadata (the image has no onnx package)."""
    m = json.load(open(path))
    m["names"] = {int(k): v for k, v in m["names"].items()}
    for k in ("attr_mask", "child_mask", "child_attr_mask"):
        m[k] = np.asarray(m[k], np.float32)
    m["layout"] = [(g, tuple(s)) for g, s in m["layout"]]
    return m


def read_ini(path: Path) -> configparser.ConfigParser:
    cp = configparser.ConfigParser(interpolation=None, delimiters=("=",), comment_prefixes=("#",))
    cp.optionxform = str  # keep key case
    cp.read(path)
    return cp


def make(factory: str, name: str, **props) -> Gst.Element:
    e = Gst.ElementFactory.make(factory, name)
    if e is None:
        raise RuntimeError(f"cannot create {factory}")
    for k, v in props.items():
        e.set_property(k.replace("_", "-"), v)
    return e


def gpointer(obj) -> int:
    """C pointer of a GObject (for passing pads to the C++ library)."""
    get = ctypes.pythonapi.PyCapsule_GetPointer
    get.restype, get.argtypes = ctypes.c_void_p, [ctypes.py_object, ctypes.c_char_p]
    return get(obj.__gpointer__, None)


class Dump:
    """Pad probe after the tracker: writes detections per input and keeps persons for the analysis scripts."""

    def __init__(self, out: Path, stems: list[str], meta: dict):
        self.out, self.stems, self.meta = out, stems, meta
        self.payload_bytes = meta["payload_size"] * 4
        self.person = next(c for c, n in meta["names"].items() if n == "person")
        self.files = {i: open(out / f"{s}_engine_dets.jsonl", "w") for i, s in enumerate(stems)}
        keys = ("frame", "scores", "boxes", "attrs", "child_conf", "child_boxes", "child_attrs", "origin")
        self.raw = {i: {k: [] for k in keys} for i in range(len(stems))}
        self.frames = Counter()
        self.objects = Counter()
        self.skipped = Counter()  # objects without this model's payload, by (class, mask_params.size)

    def __call__(self, pad, info, _=None):
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(info.get_buffer()))
        l_frame = batch.frame_meta_list
        while l_frame is not None:
            fm = pyds.NvDsFrameMeta.cast(l_frame.data)
            src, n, dets = fm.source_id, fm.frame_num, []
            l_obj = fm.obj_meta_list
            while l_obj is not None:
                o = pyds.NvDsObjectMeta.cast(l_obj.data)
                l_obj = l_obj.next
                if o.mask_params.size != self.payload_bytes:  # child objects added by attach, tracker-only boxes
                    self.skipped[(self.meta["names"].get(o.class_id, str(o.class_id)), int(o.mask_params.size))] += 1
                    continue
                b, t = o.detector_bbox_info.org_bbox_coords, o.rect_params
                box = [b.left, b.top, b.left + b.width, b.top + b.height]
                origin = ORIGIN.get(int(o.misc_obj_info[0]), "?")
                self.objects[(self.meta["names"][o.class_id], origin)] += 1
                dets.append({
                    "cls": self.meta["names"][o.class_id], "cls_id": o.class_id, "score": round(o.confidence, 4),
                    "box": [round(v, 1) for v in box], "origin": origin,
                    "track_id": None if o.object_id == UNTRACKED else int(o.object_id),
                    "track_box": [round(v, 1) for v in (t.left, t.top, t.left + t.width, t.top + t.height)],
                })
                if o.class_id == self.person:  # payload is relative to the detector box
                    d = yolo_attr_probe.decode(o.mask_params.get_mask_array(), (b.left, b.top, b.width, b.height),
                                               o.class_id, self.meta)
                    r = self.raw[src]
                    r["frame"].append(n)
                    r["origin"].append(int(origin == "full"))  # 0 mosaic, 1 full frame
                    r["scores"].append(o.confidence)
                    r["boxes"].append(box)
                    for k in ("attrs", "child_conf", "child_boxes", "child_attrs"):
                        r[k].append(d[k])
            self.files[src].write(json.dumps({"frame": n, "dets": dets}) + "\n")
            self.frames[src] = max(self.frames[src], n + 1)
            l_frame = l_frame.next
        return Gst.PadProbeReturn.OK

    def close(self, size: tuple[int, int]):
        amask, cmask = display_masks(self.meta)
        for i, stem in enumerate(self.stems):
            self.files[i].close()
            r = {k: np.asarray(v, np.float32) for k, v in self.raw[i].items()}
            r["frame"] = r["frame"].astype(np.int64)
            r["origin"] = r["origin"].astype(np.int64)
            np.savez_compressed(self.out / f"{stem}_person_raw.npz", n_frames=self.frames[i], size=list(size),
                                attr_show=amask[self.person], child_attr_show=cmask, **r)


def plan_counters(events: Path, n: int) -> dict:
    """Per-source slot and region counts from the library's per-frame plans."""
    out = defaultdict(Counter)
    for line in open(events):
        e = json.loads(line)
        c = out[e["source_id"]]
        c["frames"] += 1
        c["mosaic_slots"] += e["tensor_slot"] is not None
        c["full_slots"] += e["full_slot"] is not None
        c["frames_with_regions"] += e["regions"] > 0
        c["region_only_mosaics"] += e["status"] == "active" and e["motion_boxes"] == 0
        c["regions_total"] += e["regions"]
    return {s: dict(out[s]) for s in range(n)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/app.yaml")
    ap.add_argument("--input", action="append", required=True, help="file path or URI; repeat for more cameras")
    ap.add_argument("--output", default=None)
    ap.add_argument("--osd", choices=("on", "off"), default=None, help="override `osd` in the config")
    ap.add_argument("--mux", default=None, help="override the mux size, WIDTHxHEIGHT (use the cameras' native size)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a preprocess.ini [user-configs] key, e.g. full-frame-interval=30; repeatable")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / args.config))
    if args.mux:
        cfg["muxer"]["width"], cfg["muxer"]["height"] = map(int, args.mux.lower().split("x"))
    out = ROOT / (args.output or cfg["output"])
    (out / "plans").mkdir(parents=True, exist_ok=True)
    osd = cfg["osd"] if args.osd is None else args.osd == "on"
    stems = [Path(p).stem for p in args.input]
    meta = load_meta(ROOT / cfg["meta"])

    # Runtime preprocess config: this run's sources and log folder
    pre_cfg = read_ini(ROOT / cfg["preprocess_config"])
    pre_cfg["group-0"]["src-ids"] = ";".join(str(i) for i in range(len(args.input)))
    pre_cfg["user-configs"]["preview-dir"] = str(out / "plans")
    for item in args.set:
        key, value = item.split("=", 1)
        pre_cfg["user-configs"][key.strip()] = value.strip()
    with open(out / "preprocess.ini", "w") as f:
        pre_cfg.write(f, space_around_delimiters=False)
    component = int(pre_cfg["property"]["unique-id"])
    motion_lib = ctypes.CDLL(pre_cfg["property"]["custom-lib-path"])  # same path as nvdspreprocess: one registry
    motion_lib.vms_motion_install_probes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    motion_lib.vms_motion_install_probes.restype = ctypes.c_bool

    Gst.init(None)
    pipe = Gst.Pipeline.new("motion-ds")
    m = cfg["muxer"]
    mux = make("nvstreammux", "mux", batch_size=len(args.input), width=m["width"], height=m["height"],
               batched_push_timeout=m["batched-push-timeout"], live_source=m["live-source"], nvbuf_memory_type=2)
    pre = make("nvdspreprocess", "pre", config_file=str(out / "preprocess.ini"))
    infer = make("nvinfer", "infer", config_file_path=str(ROOT / cfg["infer_config"]))
    chain = [mux, pre, infer]
    if cfg.get("tracker_config"):
        tpath = ROOT / cfg["tracker_config"]
        t = read_ini(tpath)["tracker"]
        chain.append(make("nvtracker", "tracker", tracker_width=int(t["tracker-width"]),
                          tracker_height=int(t["tracker-height"]), gpu_id=int(t.get("gpu-id", 0)),
                          ll_lib_file=t["ll-lib-file"], ll_config_file=str(tpath.parent / t["ll-config-file"])))
    for e in chain:
        pipe.add(e)
    for a, b in zip(chain, chain[1:]):
        assert a.link(b), f"cannot link {a.name} -> {b.name}"

    def on_pad(_src, pad, queue):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if caps.get_structure(0).get_name().startswith("video/"):
            pad.link(queue.get_static_pad("sink"))

    for i, path in enumerate(args.input):
        uri = path if "://" in path else Gst.filename_to_uri(str(Path(path).resolve()))
        src = make("nvurisrcbin", f"src{i}", uri=uri, disable_audio=True)
        q = make("queue", f"q{i}")
        conv = make("nvvideoconvert", f"conv{i}", nvbuf_memory_type=2)
        caps = make("capsfilter", f"rgba{i}", caps=Gst.Caps.from_string("video/x-raw(memory:NVMM),format=RGBA"))
        for e in (src, q, conv, caps):
            pipe.add(e)
        q.link(conv)
        conv.link(caps)
        caps.get_static_pad("src").link(mux.request_pad_simple(f"sink_{i}"))
        src.connect("pad-added", on_pad, q)

    last = chain[-1]
    if osd:
        demux = make("nvstreamdemux", "demux")
        pipe.add(demux)
        last.link(demux)
        for i, stem in enumerate(stems):
            parts = [make("queue", f"oq{i}"), make("nvdsosd", f"osd{i}"), make("nvvideoconvert", f"oconv{i}"),
                     make("capsfilter", f"nv12{i}", caps=Gst.Caps.from_string("video/x-raw(memory:NVMM),format=NV12")),
                     make("nvv4l2h264enc", f"enc{i}", bitrate=int(cfg.get("bitrate", 4000000))),
                     make("h264parse", f"parse{i}"), make("qtmux", f"mp4{i}"),
                     make("filesink", f"file{i}", location=str(out / f"{stem}_ds.mp4"), sync=False)]
            for e in parts:
                pipe.add(e)
            demux.request_pad_simple(f"src_{i}").link(parts[0].get_static_pad("sink"))
            for a, b in zip(parts, parts[1:]):
                assert a.link(b), f"cannot link {a.name} -> {b.name}"
    else:
        # enable-last-sample would pin one batch, i.e. one of the few preprocess tensor buffers (tensor-buf-pool-size)
        sink = make("fakesink", "sink", sync=False, enable_last_sample=False)
        pipe.add(sink)
        last.link(sink)

    # C++ probes: drop empty preprocess tensors, decode every slot into objects and feed regions back
    if not motion_lib.vms_motion_install_probes(gpointer(pre.get_static_pad("src")),
                                                gpointer(infer.get_static_pad("src")), component):
        raise RuntimeError("vms_motion_install_probes failed")
    dump = Dump(out, stems, meta)
    a = cfg["attach"]
    tail = last.get_static_pad("src")
    tail.add_probe(Gst.PadProbeType.BUFFER, dump)  # before attach adds child objects
    tail.add_probe(Gst.PadProbeType.BUFFER, functools.partial(
        yolo_attr_probe.probe, attr_thr=a["attr_thr"], child_thr=a["child_thr"], kpt_thr=a["kpt_thr"],
        kpt_classes=set(a["kpt_classes"])), meta)

    loop, failed = GLib.MainLoop(), []

    def on_message(_bus, msg):
        if msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            failed.append(f"{msg.src.name}: {err.message} ({dbg})")
            loop.quit()

    bus = pipe.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)
    start = time.time()
    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipe.set_state(Gst.State.NULL)
    wall = time.time() - start
    dump.close((m["width"], m["height"]))
    frames = sum(dump.frames.values())
    summary = {
        "inputs": args.input, "failed": failed, "wall_s": round(wall, 1), "frames": dict(dump.frames),
        "fps_all_streams": round(frames / wall, 1),
        "objects": {f"{c}/{o}": k for (c, o), k in sorted(dump.objects.items(), key=lambda kv: -kv[1])},
        "objects_without_payload": {f"{c}/size {s}": k for (c, s), k in dump.skipped.most_common()},
        "plans": plan_counters(out / "plans" / f"events_{component}.jsonl", len(args.input)),
        "osd": osd, "config": cfg, "preprocess_overrides": args.set,
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("failed", "wall_s", "frames", "fps_all_streams", "objects",
                                              "objects_without_payload", "plans")}, indent=1))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
