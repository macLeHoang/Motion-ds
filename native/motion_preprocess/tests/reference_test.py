"""Compare compiled native implementations against the existing Python reference."""
import argparse
import subprocess
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dls.core import motion
from dls.core.regions import cap_zoom, expand_box, merge_overlapping, pack_mosaic

motion.gaussian_filter = None  # Use the same documented OpenCV fallback on both platforms.
BUILD = None


class ReferenceTest(unittest.TestCase):
    def test_geometry_matches_python_for_random_scenes(self):
        rng = np.random.default_rng(21)
        inputs, expected = [], []
        for _ in range(80):
            w, h, side, gap = 640, 360, int(rng.choice([416, 640])), int(rng.choice([0, 4, 8]))
            boxes = []
            for _ in range(int(rng.integers(1, 20))):
                x, y = int(rng.integers(w - 2)), int(rng.integers(h - 2))
                boxes.append((x, y, min(w, x + int(rng.integers(2, 100))), min(h, y + int(rng.integers(2, 130)))))
            inputs.append(" ".join(map(str, [w, h, side, side, .15, gap, 2., 1.3, len(boxes)]
                                       + [v for b in boxes for v in b])))
            crops = merge_overlapping([expand_box(b, .15, (h, w)) for b in boxes], max_growth=1.3)
            image, tiles = pack_mosaic(np.zeros((h, w, 3), np.uint8), crops, gap)
            image = cap_zoom(image, side, 2.)
            mh, mw = image.shape[:2]
            scale = min(side / mw, side / mh)
            rw, rh = round(mw * scale), round(mh * scale)
            expected.append([mw, mh, rw, rh, (side - rw) // 2, (side - rh) // 2,
                             rw / mw, rh / mh, len(tiles)]
                            + [v for p in tiles for v in (*p.region, p.x, p.y, p.w, p.h)])
        result = subprocess.run([str(BUILD / "vms_geometry_oracle")], input="\n".join(inputs) + "\n",
                                capture_output=True, text=True, check=True)
        rows = result.stdout.splitlines()
        self.assertEqual(len(rows), len(expected))
        for row, reference in zip(rows, expected):
            np.testing.assert_allclose(list(map(float, row.split())), reference, rtol=0, atol=1e-10)

    def compare_motion(self, frames, stale=0, masks=(), protect=()):
        exe = BUILD / "vms_motion_oracle"
        if not exe.exists():
            self.skipTest("OpenCV development files unavailable; native motion target not built")
        h, w = frames[0].shape
        header = [w, h, 100, 30, stale, 1, len(masks)]
        for polygon in masks:
            header += [len(polygon)] + [v for point in polygon for v in point]
        header += [len(protect)] + [v for b in protect for v in b]
        payload = (" ".join(map(str, header)) + "\n").encode() + b"".join(f.tobytes() for f in frames)
        run = subprocess.run([str(exe)], input=payload, capture_output=True, check=True)
        reference = motion.MotionDetector((h, w), stale_frames=stale, mask_polygons=masks)
        reference.protect(protect)
        rows = run.stdout.decode().splitlines()
        self.assertEqual(len(rows), len(frames))
        for frame, row in zip(frames, rows):
            boxes = reference.detect(frame)
            values = list(map(int, row.split()))
            self.assertEqual(values[0], int(reference.calibrating))
            actual = [tuple(values[i:i + 4]) for i in range(2, len(values), 4)]
            self.assertEqual(sorted(actual), sorted(boxes))

    @staticmethod
    def scene():
        rng = np.random.default_rng(7)
        return np.kron(rng.integers(60, 120, (18, 32), dtype=np.uint8), np.ones((8, 8), np.uint8))

    def test_motion_static_moving_and_persistent(self):
        bg = self.scene()
        frames = [bg] * 300
        for i in range(20):
            frame = bg.copy()
            frame[40:90, 50 + i * 3:74 + i * 3] = 250
            frames.append(frame)
        frames += [frames[-1]] * 400
        self.compare_motion(frames)

    def test_motion_stale_lighting_masks_and_protection(self):
        bg = self.scene()
        lit = bg.copy()
        lit[:, :128] = np.clip(lit[:, :128].astype(int) + 70, 0, 255)
        self.compare_motion([bg] * 300 + [lit] * 40, stale=5)
        moved = bg.copy()
        moved[40:90, 100:124] = 250
        self.compare_motion([bg] * 300 + [moved] * 10, stale=5, protect=[(100, 40, 124, 90)])
        self.compare_motion([bg] * 300 + [moved] * 10, masks=[[(.3, .2), (.6, .2), (.6, .8), (.3, .8)]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    args, rest = parser.parse_known_args()
    BUILD = Path(args.build_dir).resolve()
    unittest.main(argv=[sys.argv[0], *rest])
