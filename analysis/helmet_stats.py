"""Helmet (head child `has_helm`) confidence breakdown from infer_video.py --raw-persons output.

Only heads with child score > --child-thr count. Prints Markdown: histogram, breakdown by head box height, head score
and person score, a timeline per window, and frames worth looking at.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

BINS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def row(v: np.ndarray) -> list:
    """n, mean, p10, p50, p90, % > 0.3/0.5/0.7 of helmet probabilities."""
    if not v.size:
        return [0] + ["-"] * 7
    p = np.percentile(v, [10, 50, 90])
    return [v.size, f"{v.mean():.3f}", f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}",
            *[f"{100 * (v > t).mean():.1f}" for t in (0.3, 0.5, 0.7)]]


def table(rows: list, head: list) -> str:
    """Markdown table."""
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|",
                      *["| " + " | ".join(str(x) for x in r) + " |" for r in rows]])


def main():
    """Print the helmet report for one raw dump."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw")
    ap.add_argument("--meta", default="configs/meta.json")
    ap.add_argument("--child-thr", type=float, default=0.25)
    ap.add_argument("--window", type=int, default=750, help="timeline window in frames")
    args = ap.parse_args()
    z, meta = np.load(args.raw), json.load(open(args.meta))
    head, helm = meta["child_names"].index("head"), meta["child_attr_names"].index("has_helm")
    cap = meta["child_attr_names"].index("has_cap_hat")
    m = z["child_conf"][:, head] > args.child_thr
    f, ps, hc = z["frame"][m], z["scores"][m], z["child_conf"][m, head]
    hb = z["child_boxes"][m, head]
    hh = hb[:, 3] - hb[:, 1]
    v, vc = z["child_attrs"][m, head, helm], z["child_attrs"][m, head, cap]
    cols = ["heads", "mean", "p10", "p50", "p90", "% >0.3", "% >0.5", "% >0.7"]

    out = [f"## Helmet (`has_helm`) confidence on detected heads (head score > {args.child_thr})", "",
           f"{m.sum()} of {len(m)} person detections have a head.", "", table([row(v)], cols), ""]
    hist, _ = np.histogram(v, BINS)
    out += [table([[f"{a:.1f}-{b:.1f}", c, f"{100 * c / max(v.size, 1):.1f}"] for a, b, c in zip(BINS, BINS[1:], hist)],
                  ["has_helm", "heads", "%"]), ""]
    for name, x, edges in [("head box height (px)", hh, [0, 20, 30, 45, 70, 1e9]),
                           ("head score", hc, [0.25, 0.4, 0.5, 0.6, 0.7, 1.01]),
                           ("person score", ps, [0.25, 0.5, 0.7, 0.8, 1.01])]:
        rows = []
        for lo, hi in zip(edges, edges[1:]):
            k = (x >= lo) & (x < hi)
            rows.append([f"{lo:g}-{hi:g}" if hi < 1e9 else f"≥{lo:g}", *row(v[k])])
        out += [f"By {name}:", "", table(rows, [name, *cols]), ""]
    out += [f"`has_cap_hat` on the same heads: mean {vc.mean():.3f}, > 0.5 on {100 * (vc > 0.5).mean():.1f} %; "
            f"helmet and cap both > 0.5 on {100 * ((v > 0.5) & (vc > 0.5)).mean():.1f} %.", ""]

    rows = []
    for t0 in range(0, int(z["n_frames"]), args.window):
        k = (f >= t0) & (f < t0 + args.window)
        if k.any():
            rows.append([f"{t0}-{t0 + args.window - 1}", len(set(f[k])), *row(v[k])[:5], row(v[k])[6],
                         f"{np.median(hh[k]):.0f}"])
    out += ["Timeline:", "", table(rows, ["frames", "frames with head", *cols[:5], "% >0.5", "head h p50"]), ""]
    order = np.argsort(-v)
    pick = {"top": order[:3].tolist(), "near 0.5": np.argsort(np.abs(v - 0.5))[:3].tolist(),
            "largest heads with helm < 0.3": [i for i in np.argsort(-hh) if v[i] < 0.3][:3]}
    out += ["Frames to look at:", ""]
    out += [f"- {k}: " + ", ".join(f"frame {f[i]} (helm {v[i]:.2f}, head {hc[i]:.2f}, {hh[i]:.0f} px)" for i in idx)
            for k, idx in pick.items()]
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
