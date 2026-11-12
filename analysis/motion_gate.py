"""Motion-mosaic vs full-frame inference on one video: gate recall, object scale, matched-person attributes, extras.

Usage: motion_gate.py VIDEO_STEM --full DIR --mosaic DIR --plans DIR [--meta model/meta.json] [--helmet-frames A:B]
--full / --mosaic hold <stem>_person_raw.npz and <stem>_engine_dets.jsonl (infer_video.py / mosaic_infer.py).
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

from compare_runs import match

THR = 0.5  # person score for "confident" detections


def pct(x) -> str:
    x = np.asarray(x)
    return f"{100 * x.mean():.1f}" if x.size else "-"


def q(v, ps=(10, 50, 90)) -> str:
    v = np.asarray(v, float)
    return " / ".join(f"{np.percentile(v, p):.0f}" for p in ps) if v.size else "-"


def table(rows, head) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|",
                      *["| " + " | ".join(str(x) for x in r) + " |" for r in rows]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stem")
    ap.add_argument("--full", required=True)
    ap.add_argument("--mosaic", required=True)
    ap.add_argument("--plans", required=True)
    ap.add_argument("--source", type=int, default=None, help="source_id of this video in a multi-input plans log")
    ap.add_argument("--meta", default="configs/meta.json")
    ap.add_argument("--helmet-frames", default=None, help="A:B frame range where the person wears a helmet")
    a = ap.parse_args()
    meta = json.load(open(a.meta))
    helm = meta["child_attr_names"].index("has_helm")
    cap = meta["child_attr_names"].index("has_cap_hat")
    F = dict(np.load(Path(a.full) / f"{a.stem}_person_raw.npz"))
    M = dict(np.load(Path(a.mosaic) / f"{a.stem}_person_raw.npz"))
    plans = {e["frame_num"]: e for f in glob.glob(f"{a.plans}/events_*.jsonl") for e in map(json.loads, open(f))
             if a.source is None or e["source_id"] == a.source}
    for p in plans.values():  # plan v2: frames with only a full-frame slot
        if p["status"] != "active" and p.get("full_slot") is not None:
            p["status"] = "full frame only"
    imgsz = 768
    w = int(F["size"][0])
    out = [f"# Motion mosaic vs full frame — `{a.stem}`", ""]

    # 1. Load: which frames get inference
    st = Counter(p["status"] for p in plans.values())
    full_slots = sum(p.get("full_slot") is not None for p in plans.values())
    n = int(F["n_frames"])
    out += [f"Frames {n}: mosaic inferred {st['active']} ({100 * st['active'] / n:.1f} %), full-frame slots "
            f"{full_slots} ({st['full frame only']} of them alone), idle {st['idle']}, calibrating {st['calibrating']}.",
            ""]

    # 2. Gate: confident full-frame persons -> frame status, centre inside a crop, matched by a mosaic person
    conf = F["scores"] >= THR
    fr, bx = F["frame"][conf], F["boxes"][conf]
    status = np.array([plans.get(int(f), {}).get("status", "missing") for f in fr])
    inside = []
    for f, b in zip(fr, bx):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        tiles = plans.get(int(f), {}).get("placements", []) if plans.get(int(f), {}).get("status") == "active" else []
        inside.append(any(t["source"][0] <= cx < t["source"][2] and t["source"][1] <= cy < t["source"][3] for t in tiles))
    inside = np.array(inside)
    pairs = match(F, M)
    fi = np.array([p[0] for p in pairs], int)
    mi = np.array([p[1] for p in pairs], int)
    matched_full = np.zeros(len(F["scores"]), bool)
    matched_full[fi] = True
    rows = [[s, int((status == s).sum()), pct(inside[status == s]), pct(matched_full[conf][status == s])]
            for s in ("active", "full frame only", "idle", "calibrating", "missing") if (status == s).any()]
    rows.append(["all", int(conf.sum()), pct(inside), pct(matched_full[conf])])
    out += [f"## Full-frame persons with score ≥ {THR} (reference) — does the motion path see them?", "",
            table(rows, ["frame status", "persons", "% centre inside a motion crop", "% matched by a mosaic person (IoU ≥ 0.5)"]), ""]
    lost = conf & ~matched_full
    out += [f"Missed confident persons: {int(lost.sum())} in {len(set(F['frame'][lost].tolist()))} frames; "
            f"their full-frame score median {np.median(F['scores'][lost]):.2f}." if lost.any() else "No confident person missed.", ""]

    # 3. Extras: confident mosaic persons with no full-frame match
    mconf = M["scores"] >= THR
    matched_m = np.zeros(len(M["scores"]), bool)
    matched_m[mi] = True
    extra = mconf & ~matched_m
    out += [f"Mosaic persons with score ≥ {THR}: {int(mconf.sum())}; without a full-frame match: {int(extra.sum())} "
            f"(score median {np.median(M['scores'][extra]):.2f})." if extra.any() else
            f"Mosaic persons with score ≥ {THR}: {int(mconf.sum())}; all have a full-frame match.", ""]

    # 4. Object size in the model input, matched pairs
    fh = (F["boxes"][fi, 3] - F["boxes"][fi, 1]) * imgsz / w
    # mosaic input height per matched person: frame height x the scale of the slot it came from
    origin = M.get("origin", np.zeros(len(M["scores"]), int))  # 0 mosaic, 1 full frame (DeepStream app dumps)
    sc = np.array([plans[int(f)]["full_scale" if o == 1 else "scale"][1] for f, o in zip(M["frame"][mi], origin[mi])])
    sc = sc if len(mi) else np.zeros(0)
    mh = (M["boxes"][mi, 3] - M["boxes"][mi, 1]) * sc
    out += ["## Person height in the model input (px, p10 / p50 / p90), matched pairs", "",
            table([["full frame (scale 0.6)", q(fh)], ["motion mosaic", q(mh)],
                   ["mosaic / full", q(mh / np.maximum(fh, 1), (10, 50, 90)).replace(" ", "")]],
                  ["input", "person height"]), ""]

    # 5. Matched-person agreement and attribute shifts
    d = lambda k: np.abs(F[k][fi] - M[k][mi])  # noqa: E731
    head_both = (F["child_conf"][fi, 0] > 0.25) & (M["child_conf"][mi, 0] > 0.25)
    rows = [["matched persons", len(pairs)], ["box IoU mean", f"{np.mean([p[2] for p in pairs]):.3f}"],
            ["score mean full / mosaic", f"{F['scores'][fi].mean():.3f} / {M['scores'][mi].mean():.3f}"],
            ["posture same %", pct(F["attrs"][fi, :4].argmax(1) == M["attrs"][mi, :4].argmax(1))]]
    for c, cn in enumerate(meta["child_names"]):
        rows.append([f"{cn} present full / mosaic %", f"{pct(F['child_conf'][fi, c] > 0.25)} / {pct(M['child_conf'][mi, c] > 0.25)}"])
    out += ["## Matched persons", "", table(rows, ["", "value"]), ""]

    # 6. Helmet on matched heads (optionally only inside the helmet segment)
    segs = [("all frames", np.ones(len(fi), bool))]
    if a.helmet_frames:
        lo, hi = map(int, a.helmet_frames.split(":"))
        segs.append((f"frames {lo}-{hi} (helmet worn)", (F["frame"][fi] >= lo) & (F["frame"][fi] < hi)))
        segs.append((f"other frames", ~((F["frame"][fi] >= lo) & (F["frame"][fi] < hi))))
    rows = []
    for name, s in segs:
        m = s & head_both
        hf, hm = F["child_attrs"][fi[m], 0, helm], M["child_attrs"][mi[m], 0, helm]
        cf, cm = F["child_attrs"][fi[m], 0, cap], M["child_attrs"][mi[m], 0, cap]
        rows.append([name, int(m.sum()), f"{hf.mean():.3f} / {hm.mean():.3f}" if m.any() else "-",
                     f"{pct(hf > 0.5)} / {pct(hm > 0.5)}", f"{pct(hf > 0.3)} / {pct(hm > 0.3)}",
                     f"{pct(cf > 0.5)} / {pct(cm > 0.5)}"])
    out += ["## Helmet on matched heads (full / mosaic)", "",
            table(rows, ["frames", "heads", "has_helm mean", "has_helm > 0.5 %", "has_helm > 0.3 %", "has_cap_hat > 0.5 %"]), ""]

    # 7. All classes on active frames (thresholded detection files): per-class counts full vs mosaic
    def counts(path, frames):
        c = Counter()
        for line in open(path):
            r = json.loads(line)
            if r["frame"] in frames:
                c.update(d["cls"] for d in r["dets"])
        return c
    act = {f for f, p in plans.items() if p["status"] == "active"}
    cf = counts(Path(a.full) / f"{a.stem}_engine_dets.jsonl", act)
    cm = counts(Path(a.mosaic) / f"{a.stem}_engine_dets.jsonl", act)
    allf = counts(Path(a.full) / f"{a.stem}_engine_dets.jsonl", set(range(n)))
    keys = sorted(set(cf) | set(cm) | set(allf), key=lambda k: -(cf[k] + cm[k] + allf[k]))
    out += ["## Detections per class (score ≥ 0.25)", "",
            table([[k, allf[k], cf[k], cm[k]] for k in keys],
                  ["class", "full frame, all frames", "full frame, active frames", "mosaic (active frames)"]), ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
