#!/usr/bin/env python3
"""
Week 4 - Phase 3 helper: HSV calibration.

The plant/soil colors in RTX-rendered frames are NOT the raw displayColor values
from the USD -- tonemapping shifts them. This script samples actual pixels from a
handful of dataset frames and prints suggested HSV bounds for the green-vegetation
mask, so detect_row.py is tuned to the real render instead of a guess.

HOW IT WORKS
  - Loads a few frames from the dataset.
  - Converts to HSV.
  - Uses a permissive "greenish" pre-filter (hue in a wide green band) to grab
    candidate vegetation pixels, then reports the 5th-95th percentile H/S/V range
    of those pixels. That range becomes your starting lower/upper bound.
  - Also reports the soil pixel stats so you can see the separation.

RUN (normal Jazzy python where cv2/numpy exist -- NOT isaac's python):
    python3 ~/calibrate_hsv.py --data ~/Documents/week4_dataset

Then copy the printed LOWER_HSV / UPPER_HSV into detect_row.py.
"""

import os
import glob
import argparse
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=os.path.expanduser("~/Documents/week4_dataset"))
parser.add_argument("--n", type=int, default=8, help="how many frames to sample")
args = parser.parse_args()

frame_dir = os.path.join(args.data, "frames")
frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
if not frames:
    raise SystemExit(f"No frames found in {frame_dir}")

# sample evenly across the dataset (different rows/lighting)
idxs = np.linspace(0, len(frames) - 1, min(args.n, len(frames))).astype(int)
sample = [frames[i] for i in idxs]
print(f"Sampling {len(sample)} frames from {len(frames)} total\n")

green_pixels = []
soil_pixels = []

for f in sample:
    bgr = cv2.imread(f)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Permissive green pre-filter (OpenCV hue is 0-179; green ~ 35-90).
    # Require some saturation so we skip washed-out sky/soil.
    green_mask = (H >= 30) & (H <= 95) & (S >= 40) & (V >= 40)
    # Soil: low-ish saturation, hue in the orange/brown band (0-30) OR gray.
    soil_mask = ((H <= 25) | (H >= 160)) & (V >= 30)

    gp = hsv[green_mask]
    sp = hsv[soil_mask]
    if len(gp) > 0:
        green_pixels.append(gp)
    if len(sp) > 0:
        soil_pixels.append(sp)

if not green_pixels:
    raise SystemExit("No green pixels found with the pre-filter -- the render may "
                     "be darker/different than expected. Tell me and we widen the band.")

green = np.concatenate(green_pixels, axis=0)
soil = np.concatenate(soil_pixels, axis=0) if soil_pixels else np.zeros((1, 3))


def pct(arr, lo, hi):
    return np.percentile(arr, lo, axis=0).astype(int), np.percentile(arr, hi, axis=0).astype(int)


g_lo, g_hi = pct(green, 5, 95)
print("=== GREEN (vegetation) pixel HSV distribution ===")
print(f"  count: {len(green):,}")
print(f"  H: {green[:,0].min():3d} .. {green[:,0].max():3d}   (5-95%: {g_lo[0]}..{g_hi[0]})")
print(f"  S: {green[:,1].min():3d} .. {green[:,1].max():3d}   (5-95%: {g_lo[1]}..{g_hi[1]})")
print(f"  V: {green[:,2].min():3d} .. {green[:,2].max():3d}   (5-95%: {g_lo[2]}..{g_hi[2]})")

print("\n=== SOIL/background pixel HSV distribution (for separation check) ===")
if len(soil) > 1:
    print(f"  count: {len(soil):,}")
    print(f"  H: {soil[:,0].min():3d} .. {soil[:,0].max():3d}")
    print(f"  S: {soil[:,1].min():3d} .. {soil[:,1].max():3d}")
    print(f"  V: {soil[:,2].min():3d} .. {soil[:,2].max():3d}")

# Suggest bounds: widen the green 5-95% a little for margin, clamp to valid ranges.
lower = [max(0, g_lo[0] - 5), max(20, g_lo[1] - 20), max(20, g_lo[2] - 20)]
upper = [min(179, g_hi[0] + 5), 255, 255]

print("\n=== SUGGESTED BOUNDS for detect_row.py ===")
print(f"LOWER_HSV = np.array([{lower[0]}, {lower[1]}, {lower[2]}])")
print(f"UPPER_HSV = np.array([{upper[0]}, {upper[1]}, {upper[2]}])")
print("\nPaste those two lines into detect_row.py (replacing the defaults),")
print("then run detection. If dead-plant rows get missed, tell me -- that's the")
print("brown-vs-soil overlap and needs a second mask, not a wider green band.")
