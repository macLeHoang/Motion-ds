"""Confidence statistics of persons, their children and attributes from infer_video.py --raw-persons output.

Values are the unthresholded engine outputs (FP16 engine, NMS conf 0.25), so every person detection counts, and
children/attributes below the drawing thresholds are included. Attribute lists per child are the display lists saved
by infer_video.py (attr_show / child_attr_show), since the checkpoint masks are all ones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POSTURE = 4
PERSON_GROUPS = {"view": "view_", "age": "age_", "hair": "hair_"}  # one-of groups among person attributes
CHILD_GROUPS = {"legs": ("has_long_pants", "has_shorts", "has_skirt"),
                "foot": ("has_boots", "has_shoes", "has_sandals")}
SCORE_BINS = [0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
PERSON_BUCKETS = [(0.25, 0.5), (0.5, 0.7), (0.7, 1.01)]


def dist(v) -> dict:
    """Mean, std and percentiles of a 1-D array."""
    v = np.asarray(v, float)
    if not v.size:
        return {"n": 0}
    p = np.percentile(v, [10, 25, 50, 75, 90])
    return {"n": int(v.size), "mean": round(v.mean(), 3), "std": round(v.std(), 3), "min": round(v.min(), 3),
            "p10": round(p[0], 3), "p25": round(p[1], 3), "p50": round(p[2], 3), "p75": round(p[3], 3),
            "p90": round(p[4], 3), "max": round(v.max(), 3)}


def pct(x) -> float:
    """Percentage of True in a boolean array."""
    x = np.asarray(x)
    return round(100 * x.mean(), 1) if x.size else float("nan")


def attr_rows(values: np.ndarray, names: list, thr: float) -> dict:
    """Per-attribute distribution and share above thr, for an (n, k) array."""
    return {a: {**dist(values[:, j]), "pct_above_thr": pct(values[:, j] > thr)} for j, a in enumerate(names)}


def argmax_share(values: np.ndarray, names: list) -> dict:
    """Share of rows whose argmax is each name, and the mean winning probability."""
    if not len(values):
        return {}
    w = values.argmax(1)
    return {a: {"pct": pct(w == j), "mean_top_prob": round(float(values[w == j, j].mean()), 3) if (w == j).any() else None}
            for j, a in enumerate(names)}


def compute(z, meta: dict, child_thr: float, attr_thr: float) -> dict:
    """All statistics as a JSON-able dict."""
    attrs, cnames, cattrs = meta["attributes"], meta["child_names"], meta["child_attr_names"]
    n_frames, frame, score = int(z["n_frames"]), z["frame"], z["scores"]
    A, CC, CA = z["attrs"], z["child_conf"], z["child_attrs"]
    show, cshow = z["attr_show"].astype(bool), z["child_attr_show"].astype(bool)
    per_frame = np.bincount(frame, minlength=n_frames)
    s = {"frames": n_frames, "frames_with_person": int((per_frame > 0).sum()), "person_detections": int(len(score)),
         "persons_per_frame": {"mean": round(per_frame.mean(), 2), "mean_when_present": round(per_frame[per_frame > 0].mean(), 2)
                               if (per_frame > 0).any() else 0, "max": int(per_frame.max())},
         "thresholds": {"child": child_thr, "attr": attr_thr}}

    # Person confidence
    hist, _ = np.histogram(score, SCORE_BINS)
    s["person_score"] = {**dist(score), "pct_ge_0.5": pct(score >= 0.5), "pct_ge_0.7": pct(score >= 0.7),
                         "histogram": {f"{a:.2f}-{b:.2f}": int(c) for a, b, c in zip(SCORE_BINS, SCORE_BINS[1:], hist)}}

    # Person attributes: posture (one choice), one-of groups, then every person attribute
    s["posture"] = argmax_share(A[:, :POSTURE], attrs[:POSTURE])
    s["posture_top_prob"] = dist(A[:, :POSTURE].max(1)) if len(A) else {"n": 0}
    s["person_groups"] = {g: argmax_share(A[:, [j for j, a in enumerate(attrs) if a.startswith(p)]],
                                          [a for a in attrs if a.startswith(p)]) for g, p in PERSON_GROUPS.items()}
    idx = [j for j in range(POSTURE, len(attrs)) if show[j]]
    s["person_attrs"] = attr_rows(A[:, idx], [attrs[j] for j in idx], attr_thr)

    # Children: confidence over all persons, presence by person-score bucket, count per person
    present = CC > child_thr
    s["children_per_person"] = {str(k): pct(present.sum(1) == k) for k in range(len(cnames) + 1)}
    s["child_conf"] = {}
    for c, cn in enumerate(cnames):
        row = {**dist(CC[:, c]), "pct_present": pct(present[:, c]), "pct_gt_0.5": pct(CC[:, c] > 0.5),
               "conf_when_present": dist(CC[present[:, c], c])}
        for lo, hi in PERSON_BUCKETS:
            m = (score >= lo) & (score < hi)
            row[f"pct_present_person_{lo:.2f}-{min(hi, 1):.2f}"] = pct(present[m, c]) if m.any() else None
        s["child_conf"][cn] = row

    # Child attributes over present children only
    s["child_attrs"] = {}
    for c, cn in enumerate(cnames):
        m, idx = present[:, c], [j for j in range(len(cattrs)) if cshow[c, j]]
        s["child_attrs"][cn] = attr_rows(CA[m][:, c, idx], [cattrs[j] for j in idx], attr_thr)
    t = cnames.index("torso")
    ts, tl = cattrs.index("is_short_sleeve"), cattrs.index("is_long_sleeve")
    sl = CA[present[:, t], t]
    s["torso_sleeves"] = {"both_above_thr_pct": pct((sl[:, ts] > attr_thr) & (sl[:, tl] > attr_thr)),
                          "short_wins_pct": pct(sl[:, ts] > sl[:, tl]), "abs_diff": dist(np.abs(sl[:, ts] - sl[:, tl]))}
    lg = cnames.index("legs")
    gi = [cattrs.index(a) for a in CHILD_GROUPS["legs"]]
    s["legs_argmax"] = argmax_share(CA[present[:, lg], lg][:, gi], list(CHILD_GROUPS["legs"]))
    fi = [cattrs.index(a) for a in CHILD_GROUPS["foot"]]
    feet = [cnames.index("foot_left"), cnames.index("foot_right")]
    fa = np.concatenate([CA[present[:, f], f][:, fi] for f in feet])
    s["foot_argmax"] = argmax_share(fa, list(CHILD_GROUPS["foot"]))
    both = present[:, feet[0]] & present[:, feet[1]]
    lw, rw = CA[both, feet[0]][:, fi].argmax(1), CA[both, feet[1]][:, fi].argmax(1)
    s["feet_left_right_same_footwear_pct"] = pct(lw == rw)
    s["feet_both_present_pct"] = pct(both)
    return s


def table(rows: list, head: list) -> list:
    """Markdown table lines."""
    return ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|",
            *["| " + " | ".join(str(x) for x in r) + " |" for r in rows]]


def markdown(s: dict, video: str) -> str:
    """Readable report of compute() output."""
    thr = s["thresholds"]
    ps = s["person_score"]
    L = [f"# Person / child / attribute confidence — `{video}`", "",
         f"Engine `AIO_t13_11_11` FP16, NMS conf 0.25. Unthresholded engine values. A child counts as present when its "
         f"score > {thr['child']}; attribute columns `>thr` use {thr['attr']}.", "",
         f"- Frames: {s['frames']}, with a person: {s['frames_with_person']} "
         f"({100 * s['frames_with_person'] / max(s['frames'], 1):.1f} %)",
         f"- Person detections: {s['person_detections']}; per frame mean {s['persons_per_frame']['mean']}, "
         f"{s['persons_per_frame']['mean_when_present']} when present, max {s['persons_per_frame']['max']}", ""]
    cols = ["mean", "p10", "p25", "p50", "p75", "p90", "max"]

    L += ["## 1. Person confidence", ""]
    L += table([[ps.get(k) for k in cols] + [ps["pct_ge_0.5"], ps["pct_ge_0.7"]]], cols + ["% ≥0.5", "% ≥0.7"])
    L += ["", *table([[k, v, f"{100 * v / max(ps['n'], 1):.1f}"] for k, v in ps["histogram"].items()],
                     ["score", "detections", "%"]), ""]

    L += ["## 2. Person attributes", "", "Posture (argmax of attributes 0–3):", ""]
    L += table([[a, v["pct"], v["mean_top_prob"]] for a, v in s["posture"].items()], ["posture", "% of persons", "mean prob"])
    for g, v in s["person_groups"].items():
        L += ["", f"{g} (argmax):", ""]
        L += table([[a, x["pct"], x["mean_top_prob"]] for a, x in v.items()], [g, "% of persons", "mean prob"])
    L += ["", "Every person attribute (posture excluded):", ""]
    L += table([[a, v["mean"], v["p10"], v["p50"], v["p90"], v["pct_above_thr"]] for a, v in s["person_attrs"].items()],
               ["attribute", "mean", "p10", "p50", "p90", "% >thr"])

    L += ["", "## 3. Child confidence", "", "Over all person detections (present or not):", ""]
    buckets = [k for k in next(iter(s["child_conf"].values())) if k.startswith("pct_present_person_")]
    L += table([[cn, v["mean"], v["p10"], v["p50"], v["p90"], v["pct_present"], v["pct_gt_0.5"],
                 v["conf_when_present"].get("mean"), v["conf_when_present"].get("p50"),
                 *[v[b] for b in buckets]] for cn, v in s["child_conf"].items()],
               ["child", "mean", "p10", "p50", "p90", "% present", "% >0.5", "mean if present", "p50 if present",
                *[f"% present, person {b.split('_')[-1]}" for b in buckets]])
    L += ["", "Children present per person:", ""]
    L += table([[k, v] for k, v in s["children_per_person"].items()], ["children", "% of persons"])

    L += ["", "## 4. Child attributes (present children only)", ""]
    for cn, rows in s["child_attrs"].items():
        n = next(iter(rows.values()))["n"] if rows else 0
        L += [f"**{cn}** ({n} children):", ""]
        L += table([[a, v.get("mean"), v.get("p10"), v.get("p50"), v.get("p90"), v.get("pct_above_thr")]
                    for a, v in rows.items()], ["attribute", "mean", "p10", "p50", "p90", "% >thr"])
        L += [""]
    sl = s["torso_sleeves"]
    L += [f"- Torso: both `is_short_sleeve` and `is_long_sleeve` > {thr['attr']} on {sl['both_above_thr_pct']} % "
          f"(short wins {sl['short_wins_pct']} %, |short − long| median {sl['abs_diff'].get('p50')}).",
          "- Legs argmax: " + ", ".join(f"{a} {v['pct']} %" for a, v in s["legs_argmax"].items()),
          "- Foot argmax (both feet pooled): " + ", ".join(f"{a} {v['pct']} %" for a, v in s["foot_argmax"].items()),
          f"- Both feet present on {s['feet_both_present_pct']} % of persons; same footwear argmax left/right on "
          f"{s['feet_left_right_same_footwear_pct']} % of those.", ""]
    return "\n".join(L)


def main():
    """Load the raw dump and write <stem>_person_stats.json and .md next to it."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw", help="<stem>_person_raw.npz from infer_video.py --raw-persons")
    ap.add_argument("--meta", default="configs/meta.json")
    ap.add_argument("--child-thr", type=float, default=0.25)
    ap.add_argument("--attr-thr", type=float, default=0.5)
    args = ap.parse_args()
    raw = Path(args.raw)
    stem = raw.name.removesuffix("_person_raw.npz")
    s = compute(np.load(raw), json.load(open(args.meta)), args.child_thr, args.attr_thr)
    json.dump(s, open(raw.with_name(f"{stem}_person_stats.json"), "w"), indent=1)
    md = markdown(s, f"{stem}.mp4")
    raw.with_name(f"{stem}_person_stats.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
