"""Hardware acceptance: static/moving/mixed batches, pass-through pixels and idle inference."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np


def write_clip(path, moving):
    rng = np.random.default_rng(21)
    base = np.kron(rng.integers(60, 120, (45, 80), dtype=np.uint8), np.ones((8, 8), np.uint8))
    base = np.repeat(base[..., None], 3, axis=2)
    process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
                                "-pix_fmt", "rgb24", "-s", "640x360", "-r", "10", "-i", "pipe:0",
                                "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "0",
                                "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    try:
        for i in range(440):
            frame = base.copy()
            if moving and i >= 300:
                x = 100 + (i - 300) % 80 * 3
                frame[120:220, x:x + 40] = [240, 220, 180]
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg failed to create fixture")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--identity-engine", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sources", type=int, default=2, help="Number of alternating static/moving sources in the mixed test (2..8)")
    args = parser.parse_args()
    if not 2 <= args.sources <= 8:
        parser.error("sources must be between 2 and the default tensor capacity of 8")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    session = args.work_dir.resolve() / f"run_{time.time_ns()}"
    session.mkdir()
    static, moving = session / "static.mp4", session / "moving.mp4"
    write_clip(static, False)
    write_clip(moving, True)
    executable = args.build_dir.resolve() / "vms_motion_preview"
    mixed = [static if i % 2 == 0 else moving for i in range(args.sources)]
    for name, inputs in [("static", [static]), ("moving", [moving]), ("mixed", mixed)]:
        output = session / name
        command = [str(executable), "--config", str(args.config.resolve()), "--output", str(output),
                   "--width", "640", "--height", "360", "--gpu", str(args.gpu), "--verify-pixels"]
        for clip in inputs:
            command += ["--input", str(clip.resolve())]
        if args.identity_engine:
            command += ["--identity-engine", str(args.identity_engine.resolve())]
        run = subprocess.run(command, capture_output=True, text=True, timeout=300)
        (session / f"{name}.stdout.txt").write_text(run.stdout)
        (session / f"{name}.stderr.txt").write_text(run.stderr)
        if run.returncode:
            raise RuntimeError(f"{name} pipeline failed; inspect logs in {session}")
        summary = json.loads(run.stdout.splitlines()[-1])
        if summary["failed"] or summary["frames"] != 440 * len(inputs):
            raise AssertionError(f"{name}: lost frames: {summary}")
        if summary["tensor_samples"] != summary["active"]:
            raise AssertionError(f"{name}: incorrect tensor compaction: {summary}")
        events = [json.loads(line) for file in output.glob("events_*.jsonl") for line in file.read_text().splitlines()]
        if name == "static" and (summary["active"] != 0 or summary["idle"] == 0):
            raise AssertionError(f"idle video generated inference samples: {summary}")
        if name != "static" and summary["active"] == 0:
            raise AssertionError(f"motion generated no tensors: {summary}")
        if name == "mixed":
            if any(e["source_id"] % 2 == 0 and e["status"] == "active" for e in events):
                raise AssertionError("static camera entered a mixed inference batch")
            active_sources = {e["source_id"] for e in events if e["status"] == "active"}
            if active_sources != set(range(1, args.sources, 2)):
                raise AssertionError("moving camera lost from mixed batch")
        print(name, summary)
    print(f"Acceptance artifacts: {session}")


if __name__ == "__main__":
    main()
