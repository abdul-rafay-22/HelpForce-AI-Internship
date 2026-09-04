#!/usr/bin/env python3
"""
Week 4 - Phase 4: Baseline accuracy evaluation.

Joins the detector's predictions (detections.csv) with ground truth (manifest.csv)
and computes baseline metrics:
  - detection rate
  - row-center offset error (pixels): mean / median / p90 / max
  - average processing time
  - error sliced by drift and by lighting, so we see WHERE it breaks

Writes results.csv (per-frame joined table) and prints a summary.

RUN (normal Jazzy python):
    python3 ~/evaluate.py --data ~/Documents/week4_dataset
"""

import os
import csv
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", default=os.path.expanduser("~/Documents/week4_dataset"))
args = parser.parse_args()

man_path = os.path.join(args.data, "manifest.csv")
det_path = os.path.join(args.data, "detections.csv")
out_path = os.path.join(args.data, "results.csv")

if not os.path.exists(det_path):
    raise SystemExit(f"Missing {det_path} -- run detect_row.py batch first.")


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


manifest = {int(r["frame_id"]): r for r in load_csv(man_path)}
detections = {int(r["frame_id"]): r for r in load_csv(det_path)}

rows = []
for fid, m in manifest.items():
    d = detections.get(fid)
    if d is None:
        continue
    gt = float(m["gt_center_px"])
    detected = int(d["detected"])
    pred = float(d["pred_center_px"])
    err = abs(pred - gt) if detected else None
    rows.append({
        "frame_id": fid,
        "row_idx": int(m["row_idx"]),
        "drift_m": float(m["drift_m"]),
        "sun_intensity": float(m["sun_intensity"]),
        "gt_center_px": gt,
        "pred_center_px": pred if detected else -1,
        "detected": detected,
        "offset_error_px": round(err, 2) if err is not None else -1,
        "proc_ms": float(d["proc_ms"]),
    })

with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

n = len(rows)
det = [r for r in rows if r["detected"]]
errs = np.array([r["offset_error_px"] for r in det], dtype=float)
proc = np.array([r["proc_ms"] for r in rows], dtype=float)

print("=" * 55)
print("WEEK 4 - BASELINE ROW-DETECTION METRICS")
print("=" * 55)
print(f"Frames evaluated:      {n}")
print(f"Detection rate:        {len(det)}/{n} = {100*len(det)/n:.1f}%")
print()
print("Offset error (detected frames only), pixels:")
print(f"  mean:                {errs.mean():.1f}")
print(f"  median:              {np.median(errs):.1f}")
print(f"  p90:                 {np.percentile(errs,90):.1f}")
print(f"  max:                 {errs.max():.1f}")
print()
print(f"Processing time:       {proc.mean():.2f} ms/frame ({1000/proc.mean():.0f} FPS)")

print("\nError by |drift| (how far the camera is off the row):")
print(f"  {'|drift| m':>10} {'frames':>7} {'mean err':>9} {'median':>8}")
for d in sorted(set(abs(r["drift_m"]) for r in det)):
    sub = [r["offset_error_px"] for r in det if abs(r["drift_m"]) == d]
    print(f"  {d:>10.2f} {len(sub):>7} {np.mean(sub):>9.1f} {np.median(sub):>8.1f}")

print("\nError by lighting (sun intensity):")
print(f"  {'sun':>10} {'frames':>7} {'mean err':>9} {'median':>8}")
for s in sorted(set(r["sun_intensity"] for r in det)):
    sub = [r["offset_error_px"] for r in det if r["sun_intensity"] == s]
    print(f"  {s:>10.0f} {len(sub):>7} {np.mean(sub):>9.1f} {np.median(sub):>8.1f}")

print("\nError by row index:")
print(f"  {'row':>10} {'frames':>7} {'mean err':>9} {'median':>8}")
for ri in sorted(set(r["row_idx"] for r in det)):
    sub = [r["offset_error_px"] for r in det if r["row_idx"] == ri]
    print(f"  {ri:>10} {len(sub):>7} {np.mean(sub):>9.1f} {np.median(sub):>8.1f}")

print(f"\nPer-frame results -> {out_path}")

print("\n" + "-" * 55)
mean_err = errs.mean()
if mean_err < 30:
    print(f"Mean error {mean_err:.0f}px is tight -- detector tracks the row well.")
elif mean_err < 80:
    print(f"Mean error {mean_err:.0f}px is moderate -- usable but check high-drift rows.")
else:
    print(f"Mean error {mean_err:.0f}px is large -- detector likely latches onto the")
    print("wrong row on multi-row frames. Worth fixing before Phase 5.")
