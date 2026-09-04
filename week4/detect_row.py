#!/usr/bin/env python3
"""
Week 4 - Phase 3: Row-detection pipeline (classical CV).

Approach: HSV-threshold the green vegetation, then estimate the row-center column.
Instead of a single global centroid (which drifts when plants are unevenly spread),
we estimate the row center per image-row (scanline) in the lower half of the frame
and take a robust average -- this tracks the vanishing-point column of the row the
camera is following, which is what a controller needs.

Outputs a detected row-center pixel (x) and a detection flag.

TWO MODES:
  batch  -- run over the synthetic dataset, write detections.csv (for Phase 4)
  live   -- subscribe to /robot/camera/image_raw, print/overlay in real time
            (for the deliverable #5 screen recording)

RUN (batch, normal Jazzy python with cv2/numpy):
    python3 ~/detect_row.py batch --data ~/Documents/week4_dataset

RUN (live, Jazzy sourced, Isaac Sim playing & publishing the camera):
    python3 ~/detect_row.py live
"""

import os
import sys
import glob
import csv
import time
import argparse
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# HSV BOUNDS -- REPLACE these with the output of calibrate_hsv.py.
# These defaults are a starting guess for RTX-rendered green crops; they will
# almost certainly need the calibrated values to perform well.
# ---------------------------------------------------------------------------
LOWER_HSV = np.array([42, 75, 49])
UPPER_HSV = np.array([66, 255, 255])

# Fraction of the image height (from the bottom) to analyze. The near field is
# where the row is widest / most reliable; the far field converges to a point.
ROI_BOTTOM_FRAC = 0.55

MIN_MASK_PIXELS = 300      # min green pixels in ROI to call it a detection
SMOOTH_WIN = 61            # column-histogram smoothing window (px, odd)
PEAK_HALF_WIN = 90         # +/- px around the chosen peak used for centroid refine


def _smooth(profile, win):
    if win <= 1:
        return profile
    k = np.ones(win, dtype=float) / win
    return np.convolve(profile, k, mode="same")


def detect_row_center(bgr, prior_cx=None):
    """
    Column-histogram row detector.

    Instead of averaging ALL green columns across the width (which gets dragged
    toward neighbouring rows when the camera drifts), we build a 1D column-
    intensity profile of the green mask over the near-field ROI. Each visible row
    is a peak. We pick the peak nearest the expected row position -- prior_cx if
    given (temporal tracking), otherwise image center -- then refine with a local
    centroid around that peak.

    Returns (cx, mask, debug):
      cx    = detected row-center x pixel (float) or None
      mask  = binary green mask (for overlay)
      debug = dict (chosen peak, profile stats)
    """
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    y0 = int(h * (1.0 - ROI_BOTTOM_FRAC))
    roi = mask[y0:h, :]

    debug = {"y0": y0}
    if int(roi.sum() // 255) < MIN_MASK_PIXELS:
        return None, mask, debug

    # 1D column profile: how much green in each column of the ROI.
    profile = roi.sum(axis=0).astype(float) / 255.0
    prof_s = _smooth(profile, SMOOTH_WIN)
    if prof_s.max() <= 0:
        return None, mask, debug

    # Candidate peaks: local maxima above a fraction of the global max.
    thresh = 0.35 * prof_s.max()
    peaks = []
    for x in range(1, w - 1):
        if prof_s[x] >= thresh and prof_s[x] >= prof_s[x - 1] and prof_s[x] > prof_s[x + 1]:
            peaks.append(x)
    if not peaks:
        peaks = [int(np.argmax(prof_s))]

    # Choose the peak nearest the expected row column.
    expected = prior_cx if prior_cx is not None else (w / 2.0)
    chosen = min(peaks, key=lambda px: abs(px - expected))

    # Refine: centroid of the raw profile in a window around the chosen peak.
    lo = max(0, int(chosen - PEAK_HALF_WIN))
    hi = min(w, int(chosen + PEAK_HALF_WIN))
    seg = profile[lo:hi]
    if seg.sum() > 0:
        cols = np.arange(lo, hi, dtype=float)
        cx = float((cols * seg).sum() / seg.sum())
    else:
        cx = float(chosen)

    debug.update({"cx": cx, "peaks": peaks, "chosen": chosen,
                  "n_peaks": len(peaks), "expected": expected})
    return cx, mask, debug


def make_overlay(bgr, cx, mask):
    ov = bgr.copy()
    # tint mask green
    green = np.zeros_like(bgr)
    green[:, :, 1] = mask
    ov = cv2.addWeighted(ov, 1.0, green, 0.4, 0)
    if cx is not None:
        cv2.line(ov, (int(cx), 0), (int(cx), bgr.shape[0]), (0, 0, 255), 2)
    # draw image center for reference
    cv2.line(ov, (bgr.shape[1] // 2, 0), (bgr.shape[1] // 2, bgr.shape[0]),
             (255, 255, 0), 1)
    return ov


# ---------------------------------------------------------------------------
# BATCH MODE
# ---------------------------------------------------------------------------
def run_batch(args):
    frame_dir = os.path.join(args.data, "frames")
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not frames:
        raise SystemExit(f"No frames in {frame_dir}")

    overlay_dir = os.path.join(args.data, "detect_overlay")
    os.makedirs(overlay_dir, exist_ok=True)

    out_csv = os.path.join(args.data, "detections.csv")
    n_overlay = 0
    rows = []
    for i, f in enumerate(frames):
        bgr = cv2.imread(f)
        t0 = time.perf_counter()
        cx, mask, dbg = detect_row_center(bgr)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        frame_id = int(os.path.splitext(os.path.basename(f))[0])
        rows.append({
            "frame_id": frame_id,
            "detected": int(cx is not None),
            "pred_center_px": round(cx, 2) if cx is not None else -1,
            "n_peaks": dbg.get("n_peaks", 0),
            "proc_ms": round(dt_ms, 2),
        })
        if n_overlay < args.overlays:
            cv2.imwrite(os.path.join(overlay_dir, f"{frame_id:04d}.png"),
                        make_overlay(bgr, cx, mask))
            n_overlay += 1

    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    det = sum(r["detected"] for r in rows)
    avg_ms = np.mean([r["proc_ms"] for r in rows])
    print(f"Processed {len(rows)} frames")
    print(f"Detection rate: {det}/{len(rows)} = {100*det/len(rows):.1f}%")
    print(f"Avg processing time: {avg_ms:.2f} ms/frame ({1000/avg_ms:.0f} FPS)")
    print(f"Detections -> {out_csv}")
    print(f"Overlays ({n_overlay}) -> {overlay_dir}")
    print("\nNext: run evaluate.py to score pred_center_px against gt_center_px.")


# ---------------------------------------------------------------------------
# LIVE MODE (ROS2)
# ---------------------------------------------------------------------------
def run_live(args):
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    class RowDetector(Node):
        def __init__(self):
            super().__init__("row_detector")
            self.bridge = CvBridge()
            self.sub = self.create_subscription(
                Image, "/robot/camera/image_raw", self.cb, 10)
            self.get_logger().info("Subscribed to /robot/camera/image_raw")

        def cb(self, msg):
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            cx, mask, dbg = detect_row_center(bgr)
            ov = make_overlay(bgr, cx, mask)
            offset = (cx - bgr.shape[1] / 2) if cx is not None else float("nan")
            txt = f"cx={cx:.0f} off={offset:+.0f}px" if cx is not None else "NO DETECTION"
            cv2.putText(ov, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2)
            cv2.imshow("row detection (live)", ov)
            cv2.waitKey(1)

    rclpy.init()
    node = RowDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)

    pb = sub.add_parser("batch")
    pb.add_argument("--data", default=os.path.expanduser("~/Documents/week4_dataset"))
    pb.add_argument("--overlays", type=int, default=30)

    pl = sub.add_parser("live")

    args = p.parse_args()
    if args.mode == "batch":
        run_batch(args)
    elif args.mode == "live":
        run_live(args)
