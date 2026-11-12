"""Fire / smoke detection timeline of several runs on one video (detection jsonl files).

Usage: fire_timeline.py --fps 29.97 label=path/<stem>_engine_dets.jsonl [label=...]
The first run is the reference (e.g. full-frame inference on every frame). For each class: detections, frames with a
detection, first detection (and delay vs the reference), share of 1-second windows with a detection inside the
reference's active span, score median / max, and a timeline per `--bin` seconds.
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def load(path: str, classes: list[str], min_score: float, held: bool) -> dict:
    """class -> {frame: max score} and the number of frames in the file; `held` keeps re-attached copies."""
    out, frames = {c: {} for c in classes}, 0
    for line in open(path):
        r = json.loads(line)
        frames = max(frames, r["frame"] + 1)
        for d in r["dets"]:
            if d["cls"] in out and d["score"] >= min_score and (held or d.get("origin") != "held"):
                out[d["cls"]][r["frame"]] = max(out[d["cls"]].get(r["frame"], 0), d["score"])
    return out, frames


def table(rows, head) -> str:
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|",
                      *["| " + " | ".join(str(x) for x in r) + " |" for r in rows]])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="label=path to a detection jsonl; the first is the reference")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--classes", nargs="+", default=["fire", "smoke"])
    ap.add_argument("--score", type=float, default=0.25)
    ap.add_argument("--bin", type=float, default=10.0, help="timeline bin in seconds")
    ap.add_argument("--include-held", action="store_true",
                    help="count held full-frame detections (what a rule after the tracker sees), not only new ones")
    a = ap.parse_args()
    runs = [(r.split("=", 1)[0], *load(r.split("=", 1)[1], a.classes, a.score, a.include_held)) for r in a.runs]
    n = max(f for _, _, f in runs)
    out = []
    for c in a.classes:
        ref = runs[0][1][c]
        span = (min(ref), max(ref)) if ref else None
        windows = None
        if span:
            w0, w1 = int(span[0] / a.fps), int(span[1] / a.fps)
            windows = np.arange(w0, w1 + 1)
        rows = []
        for label, dets, _ in runs:
            d = dets[c]
            fr = np.array(sorted(d))
            sc = np.array([d[f] for f in fr]) if len(fr) else np.zeros(0)
            first = f"{fr[0]} ({fr[0] / a.fps:.1f} s)" if len(fr) else "-"
            delay = f"{(fr[0] - span[0]) / a.fps:+.1f} s" if len(fr) and span else "-"
            cover = "-"
            if windows is not None and len(fr):
                hit = set((fr / a.fps).astype(int))
                cover = f"{100 * np.mean([w in hit for w in windows]):.0f} %"
            rows.append([label, len(fr), first, delay, cover,
                         f"{np.median(sc):.2f} / {sc.max():.2f}" if len(sc) else "-"])
        out += [f"## {c}", "", table(rows, ["run", "frames with a detection", "first detection", "vs reference",
                                           "1-s windows covered (reference span)", "score median / max"]), ""]
        bins = np.arange(0, n / a.fps + a.bin, a.bin)
        tl = [[f"{bins[i]:.0f}-{bins[i + 1]:.0f} s"] +
              [sum(bins[i] <= f / a.fps < bins[i + 1] for f in dets[c]) for _, dets, _ in runs]
              for i in range(len(bins) - 1)]
        tl = [r for r in tl if any(r[1:])]
        out += [f"Frames with {c} per {a.bin:.0f} s:", "", table(tl, ["time"] + [l for l, _, _ in runs]), ""]
    print("\n".join(out))


if __name__ == "__main__":
    main()
