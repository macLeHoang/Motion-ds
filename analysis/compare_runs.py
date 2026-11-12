"""Compare person/child/helmet outputs of several runs on the same video (raw dumps from --raw-persons / ds_infer.py).

Usage: compare_runs.py --meta model/meta.json label=path/to/<stem>_person_raw.npz [label=...]
The first run is the reference. Prints per-run statistics, then per-frame agreement of each run with the reference:
persons matched by IoU >= 0.5 (highest score first), score / posture / child / helmet differences on matched pairs.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

CHILD_THR, ATTR_THR = 0.25, 0.5


def iou(a, b):
    """IoU between xyxy boxes a (n, 4) and b (m, 4)."""
    lt, rb = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).clip(0).prod(-1)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def stats(z, helm: int, cap: int) -> list:
    """Per-run summary row."""
    s, cc, ca = z["scores"], z["child_conf"], z["child_attrs"]
    head = cc[:, 0] > CHILD_THR
    h = ca[head, 0, helm]
    return [len(s), f"{s.mean():.3f}", f"{100 * (s >= 0.5).mean():.1f}",
            *[f"{100 * (cc[:, c] > CHILD_THR).mean():.1f}" for c in range(cc.shape[1])],
            f"{h.mean():.3f}", f"{100 * (h > 0.3).mean():.1f}", f"{100 * (h > ATTR_THR).mean():.1f}",
            f"{ca[head, 0, cap].mean():.3f}"]


def match(a, b) -> list:
    """Matched (i, j, iou) person pairs per frame, a = reference."""
    pairs = []
    fb = {f: np.nonzero(b["frame"] == f)[0] for f in np.unique(b["frame"])}
    for f in np.unique(a["frame"]):
        ia, ib = np.nonzero(a["frame"] == f)[0], fb.get(f, np.array([], int))
        if not len(ib):
            continue
        m = iou(a["boxes"][ia], b["boxes"][ib])
        used = set()
        for k in np.argsort(-a["scores"][ia]):
            free = [j for j in range(len(ib)) if j not in used]
            if not free:
                break
            j = max(free, key=lambda j: m[k, j])
            if m[k, j] >= 0.5:
                used.add(j)
                pairs.append((ia[k], ib[j], m[k, j]))
    return pairs


def agree(a, b, helm: int) -> list:
    """Agreement row of run b against reference a, at the frame shift (-2..2) that matches the most persons."""
    shifts = {sh: match(a, {**b, "frame": b["frame"] + sh}) for sh in range(-2, 3)}
    shift = max(shifts, key=lambda sh: len(shifts[sh]))  # decoders can disagree on the first frame(s)
    p = shifts[shift]
    i, j, ious = (np.array(x) for x in zip(*p)) if p else (np.array([], int),) * 3
    ds = np.abs(a["scores"][i] - b["scores"][j])
    post = (a["attrs"][i, :4].argmax(1) == b["attrs"][j, :4].argmax(1))
    pa, pb = a["child_conf"][i] > CHILD_THR, b["child_conf"][j] > CHILD_THR
    both = pa[:, 0] & pb[:, 0]
    ha, hb = a["child_attrs"][i, 0, helm][both], b["child_attrs"][j, 0, helm][both]
    ua, ub = np.setdiff1d(np.arange(len(a["scores"])), i), np.setdiff1d(np.arange(len(b["scores"])), j)
    return [f"{shift:+d}", len(p), f"{100 * len(p) / max(len(a['scores']), 1):.1f}", f"{100 * len(p) / max(len(b['scores']), 1):.1f}",
            f"{ious.mean():.3f}", f"{ds.mean():.4f}", f"{np.percentile(ds, 95):.3f}" if len(ds) else "-",
            f"{100 * post.mean():.1f}", f"{100 * (pa == pb).mean():.1f}",
            f"{np.abs(a['child_conf'][i] - b['child_conf'][j]).mean():.4f}",
            f"{np.abs(ha - hb).mean():.4f}", f"{np.abs(ha - hb).max():.3f}" if both.any() else "-",
            f"{100 * ((ha > ATTR_THR) == (hb > ATTR_THR)).mean():.1f}",
            f"{np.median(a['scores'][ua]):.2f}" if len(ua) else "-", f"{np.median(b['scores'][ub]):.2f}" if len(ub) else "-"]


def table(rows, head) -> str:
    """Markdown table."""
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|",
                      *["| " + " | ".join(str(x) for x in r) + " |" for r in rows]])


def main():
    """Print both tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="configs/meta.json")
    ap.add_argument("runs", nargs="+", help="label=path to <stem>_person_raw.npz; the first is the reference")
    args = ap.parse_args()
    meta = json.load(open(args.meta))
    ca = meta["child_attr_names"]
    helm, cap = ca.index("has_helm"), ca.index("has_cap_hat")
    runs = [(r.split("=", 1)[0], dict(np.load(r.split("=", 1)[1]))) for r in args.runs]
    short = {"head": "head", "torso": "torso", "legs": "legs", "foot_left": "FL", "foot_right": "FR"}
    print(table([[k, int(z["n_frames"]), *stats(z, helm, cap)] for k, z in runs],
                ["run", "frames", "persons", "score mean", "% ≥0.5",
                 *[f"{short.get(c, c)} %" for c in meta["child_names"]], "helm mean", "helm >0.3 %", "helm >0.5 %",
                 "cap mean"]))
    ref_name, ref = runs[0]
    print(f"\nAgreement with `{ref_name}` (persons matched per frame at IoU ≥ 0.5):\n")
    print(table([[k, *agree(ref, z, helm)] for k, z in runs[1:]],
                ["run", "frame shift", "matched", "% of ref", "% of run", "box IoU", "abs Δscore mean", "abs Δscore p95",
                 "posture same %", "child presence same %", "abs Δchild conf", "abs Δhelm mean", "abs Δhelm max",
                 "helm>0.5 same %", "unmatched ref score p50", "unmatched run score p50"]))


if __name__ == "__main__":
    main()
