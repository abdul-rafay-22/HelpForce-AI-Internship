#!/usr/bin/env python3
"""
Week 4 - Phase 2: Synthetic validation dataset generator (Isaac Sim 6.0.1)

Generates a labeled validation set of camera frames from the row-crop field.
Ground truth (row-center pixel offset) is computed by projecting the known
3D row centerline through the camera's real view/projection matrices -- no
manual labeling.

RUN (from Isaac Sim python environment):
    cd ~/isaacsim/_build/linux-x86_64/release
    ./python.sh ~/generate_dataset.py

Output (in --out dir, default ~/Documents/week4_dataset):
    frames/0000.png ...          RGB captures
    manifest.csv                 frame_id, row_idx, row_y, cam_x, cam_y, cam_z,
                                 sun_intensity, gt_center_px, gt_valid, img_w, img_h
    overlay_preview/0000.png     first N frames with GT centerline drawn (sanity check)
"""

import os
import csv
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# SimulationApp MUST be created before importing any omni.* / pxr modules.
# ---------------------------------------------------------------------------
from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--usd", default=os.path.expanduser("~/Downloads/week4_field.usd"),
                    help="Path to the field USD (adjust if yours differs)")
parser.add_argument("--out", default=os.path.expanduser("~/Documents/week4_dataset"))
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--positions", type=int, default=12, help="camera positions along each row")
parser.add_argument("--preview", type=int, default=12, help="how many overlay previews to write")
parser.add_argument("--headless", action="store_true", default=True)
args = parser.parse_args()

sim_app = SimulationApp({"headless": args.headless, "renderer": "RayTracedLighting"})

# ---- now safe to import omni / pxr ----
import omni.usd
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, Gf, Sdf

# Field layout (must match week4_field.usd)
ROW_SPACING = 1.8
N_ROWS = 5
ROW_Y = [(-(N_ROWS - 1) / 2 + i) * ROW_SPACING for i in range(N_ROWS)]  # -3.6 .. 3.6
ROW_X_START, ROW_X_END = -6.0, 6.0
CAM_HEIGHT = 1.0           # camera height above ground (m) - approx Carter camera height
CAM_PITCH_TARGET_DZ = 0.3  # look at a point 0.3m above ground ahead
LOOK_AHEAD = 4.0           # how far ahead (m) the camera aims
SUN_LEVELS = [1200.0, 1600.0, 2200.0]  # lighting variation

os.makedirs(os.path.join(args.out, "frames"), exist_ok=True)
os.makedirs(os.path.join(args.out, "overlay_preview"), exist_ok=True)

# ---------------------------------------------------------------------------
# Open the field stage
# ---------------------------------------------------------------------------
ctx = omni.usd.get_context()
ok = ctx.open_stage(args.usd)
if not ok:
    print(f"[FATAL] could not open {args.usd}")
    sim_app.close()
    raise SystemExit(1)
stage = ctx.get_stage()

# The saved field file may contain a robot + Action Graph (with a dead LiDAR
# node that spams warnings). The dataset uses its own CaptureCam, so remove any
# robot/graph prims to keep the capture scene clean and quiet.
for junk in ["/World/carter_v1", "/World/ActionGraph", "/World/PhysicsGroundPlane"]:
    if stage.GetPrimAtPath(junk).IsValid():
        stage.RemovePrim(junk)
        print(f"[cleanup] removed {junk}")

# Grab the sun light so we can vary intensity
sun_prim = stage.GetPrimAtPath("/World/SunLight")
sun_intensity_attr = sun_prim.GetAttribute("inputs:intensity") if sun_prim and sun_prim.IsValid() else None

# ---------------------------------------------------------------------------
# Create a dedicated capture camera
# ---------------------------------------------------------------------------
cam_path = "/World/CaptureCam"
if stage.GetPrimAtPath(cam_path).IsValid():
    stage.RemovePrim(cam_path)
cam_prim = UsdGeom.Camera.Define(stage, cam_path)
# 24mm-ish lens on a 36mm-ish sensor -> ~60 deg horizontal FOV, reasonable for a robot cam
cam_prim.CreateFocalLengthAttr(24.0)
cam_prim.CreateHorizontalApertureAttr(36.0)
cam_prim.CreateVerticalApertureAttr(36.0 * args.height / args.width)
cam_prim.CreateClippingRangeAttr(Gf.Vec2f(0.05, 1000.0))
cam_xform = UsdGeom.Xformable(cam_prim.GetPrim())

# Render product bound to our camera
rp = rep.create.render_product(cam_path, (args.width, args.height))
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annot.attach(rp)


def set_camera_pose(eye, target, up=Gf.Vec3d(0, 0, 1)):
    """Position the camera via a look-at, using USD's SetLookAt on the xform."""
    cam_xform.ClearXformOpOrder()
    m = Gf.Matrix4d()
    m.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), up)
    # SetLookAt gives a view matrix; camera transform is its inverse
    cam_world = m.GetInverse()
    op = cam_xform.AddTransformOp()
    op.Set(cam_world)


def world_to_pixel(world_pt, width, height):
    """Project a world point to pixel coords using the camera's live matrices."""
    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    cam_world_xf = xcache.GetLocalToWorldTransform(cam_prim.GetPrim())
    view = cam_world_xf.GetInverse()
    gf_cam = UsdGeom.Camera(cam_prim.GetPrim())
    cam_data = gf_cam.GetCamera(Usd.TimeCode.Default())
    proj = cam_data.frustum.ComputeProjectionMatrix()

    p = Gf.Vec4d(world_pt[0], world_pt[1], world_pt[2], 1.0)
    p_cam = p * view           # world -> camera (view), USD row-vector convention
    p_clip = p_cam * proj      # camera -> clip, USD row-vector convention
    if abs(p_clip[3]) < 1e-9:
        return None, False
    ndc_x = p_clip[0] / p_clip[3]
    ndc_y = p_clip[1] / p_clip[3]
    # behind camera or outside frustum depth
    if p_clip[3] <= 0:
        return None, False
    px = (ndc_x * 0.5 + 0.5) * width
    py = (1.0 - (ndc_y * 0.5 + 0.5)) * height
    valid = (0 <= px < width) and (0 <= py < height)
    return (float(px), float(py)), valid


# ---------------------------------------------------------------------------
# Main capture loop
# ---------------------------------------------------------------------------
manifest_rows = []
frame_id = 0
preview_written = 0

x_positions = np.linspace(ROW_X_START, ROW_X_END - LOOK_AHEAD, args.positions)

try:
    import cv2
    have_cv2 = True
except Exception:
    have_cv2 = False
    print("[warn] cv2 not available in this env; overlay previews will be skipped")

import random
random.seed(1234)

# Lateral drift (m) applied to the camera, sideways off the row it is following.
# This makes the row appear left/right of image center so ground truth varies and
# a real detector is actually tested (a constant "output center" detector fails).
DRIFT_CHOICES = [-0.6, -0.35, -0.15, 0.0, 0.15, 0.35, 0.6]

for row_idx, row_y in enumerate(ROW_Y):
    for cam_x in x_positions:
        for sun in SUN_LEVELS:
            # set lighting
            if sun_intensity_attr:
                sun_intensity_attr.Set(float(sun))

            drift = random.choice(DRIFT_CHOICES)
            cam_y = float(row_y) + drift
            # camera sits off to the side by `drift`, but still aims straight ahead
            # (down +X at its own y), so the row appears off-center in the frame
            eye = (float(cam_x), cam_y, CAM_HEIGHT)
            target = (float(cam_x + LOOK_AHEAD), cam_y, CAM_PITCH_TARGET_DZ)
            set_camera_pose(eye, target)

            # let the renderer settle then capture
            rep.orchestrator.step(rt_subframes=8)

            rgb = rgb_annot.get_data()  # HxWx4 uint8
            if rgb is None or rgb.size == 0:
                print(f"[warn] empty frame at row {row_idx} x {cam_x:.2f}")
                continue
            rgb = rgb[:, :, :3]

            # ground-truth: project the ROW centerline point ahead (the row is at
            # row_y; the camera is offset by drift, so this projects off-center)
            gt_world = (cam_x + LOOK_AHEAD, float(row_y), CAM_PITCH_TARGET_DZ)
            gt_px, gt_valid = world_to_pixel(gt_world, args.width, args.height)
            gt_center_px = round(gt_px[0], 2) if gt_px else -1

            fname = f"{frame_id:04d}.png"
            fpath = os.path.join(args.out, "frames", fname)
            if have_cv2:
                cv2.imwrite(fpath, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                # sanity overlay for first few frames
                if preview_written < args.preview and gt_px:
                    ov = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
                    cx = int(gt_px[0])
                    cv2.line(ov, (cx, 0), (cx, args.height), (0, 0, 255), 2)
                    cv2.imwrite(os.path.join(args.out, "overlay_preview", fname), ov)
                    preview_written += 1
            else:
                from PIL import Image
                Image.fromarray(rgb).save(fpath)

            manifest_rows.append({
                "frame_id": frame_id,
                "row_idx": row_idx,
                "row_y": round(row_y, 3),
                "cam_x": round(float(cam_x), 3),
                "cam_y": round(cam_y, 3),
                "cam_z": CAM_HEIGHT,
                "drift_m": round(drift, 3),
                "sun_intensity": sun,
                "gt_center_px": gt_center_px,
                "gt_offset_from_center_px": round(gt_center_px - args.width / 2, 2) if gt_px else -9999,
                "gt_valid": int(bool(gt_valid)),
                "img_w": args.width,
                "img_h": args.height,
            })
            frame_id += 1
            print(f"[{frame_id}] row{row_idx} x={cam_x:.2f} drift={drift:+.2f} sun={sun:.0f} gt_px={gt_center_px} valid={gt_valid}")

# ---------------------------------------------------------------------------
# Write manifest
# ---------------------------------------------------------------------------
man_path = os.path.join(args.out, "manifest.csv")
with open(man_path, "w", newline="") as f:
    if manifest_rows:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

print(f"\nDONE. {frame_id} frames -> {args.out}")
print(f"Manifest: {man_path}")
print(f"Overlay previews (check these first!): {os.path.join(args.out, 'overlay_preview')}")

sim_app.close()
