"""
setup_week7_robot_and_scene.py

RUN THIS INSIDE ISAAC SIM, NOT FROM A TERMINAL:
  Window > Script Editor > paste this in > Ctrl+Enter (or the Run button)

It requires omni/isaacsim APIs (Nucleus asset resolution, live stage context)
that only exist inside the running app -- that's why this is a separate
script from generate_week7_field_curved_uneven.py, which only needs plain
usd-core and runs standalone in your env_isaacsim venv.

WHAT THIS DOES:
  1. Opens week7_field_curved_uneven.usd as your active stage (adjust
     FIELD_USD_PATH below if you put it somewhere other than next to this
     script / your home dir).
  2. References the Carter v1 robot at /World/carter_v1.
  3. Sets its start transform to the pose the field generator printed
     (default below is what a fresh run with the default SEED=42 printed --
     RE-CHECK this against your own script's console output, it changes if
     you touch the terrain/curve constants).
  4. Adds a Camera prim as a child of the robot at a reasonable front-facing
     mount offset -- VERIFY this visually against where your working Week 6
     camera was mounted; I don't know Carter's exact chassis link name so
     this parents directly under /World/carter_v1 rather than guessing a
     sub-link.
  5. Adds /World/PhysicsScene and /World/DistantLight (matching your
     documented working Week 6 fix: ~3000 intensity).

WHAT THIS DOES NOT DO -- you rebuild these by hand in the ActionGraph editor,
same as Week 6 (see checklist below the code, straight from your own working
Week 6 recipe):
  - Camera Helper node -> /robot/camera/image_raw
  - Compute Odometry -> Publish Odometry -> /odom
  - Subscribe Twist /cmd_vel -> Break 3-Vector -> Differential Controller ->
    Articulation Controller
Scripting OmniGraph blind, without being able to test it here, is a worse bet
than you rebuilding it with a recipe you've already debugged and confirmed
works.
"""

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf

# ---------------------------------------------------------------------------
# CONFIG -- check these against your actual setup before running
# ---------------------------------------------------------------------------

FIELD_USD_PATH = "/home/abdulography/Documents/week_7/week7_field_curved_uneven.usd"

CARTER_RELATIVE_PATH = "/Isaac/Robots/Carter/carter_v1.usd"
ROBOT_PRIM_PATH = "/World/carter_v1"

# From the field generator's printed output (SEED=42, default constants).
# RE-RUN the generator and re-check this if you change any field constants.
ROBOT_START_POSE = Gf.Vec3d(0.000, 0.2518, 6.000)

CAMERA_PRIM_PATH = ROBOT_PRIM_PATH + "/camera"
CAMERA_LOCAL_OFFSET = Gf.Vec3d(0.0, 0.35, 0.3)  # up + forward from robot origin, TUNE visually

LIGHT_INTENSITY = 3000.0


def get_assets_root():
    try:
        from isaacsim.core.utils.nucleus import get_assets_root_path
    except ImportError:
        from omni.isaac.core.utils.nucleus import get_assets_root_path  # older Isaac Sim versions
    root = get_assets_root_path()
    if root is None:
        raise RuntimeError("Could not find Isaac Sim assets root (Nucleus). Check Isaac Utils > Nucleus Check.")
    return root


def main():
    usd_context = omni.usd.get_context()
    usd_context.open_stage(FIELD_USD_PATH)
    stage = usd_context.get_stage()

    # --- Robot ---
    assets_root = get_assets_root()
    carter_usd = assets_root + CARTER_RELATIVE_PATH
    robot_prim = stage.DefinePrim(ROBOT_PRIM_PATH, "Xform")
    robot_prim.GetReferences().AddReference(carter_usd)
    xf = UsdGeom.Xformable(robot_prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(ROBOT_START_POSE)
    print(f"Referenced Carter from {carter_usd} at {ROBOT_PRIM_PATH}, pose {ROBOT_START_POSE}")

    # --- Camera ---
    camera = UsdGeom.Camera.Define(stage, CAMERA_PRIM_PATH)
    cam_xf = UsdGeom.Xformable(camera)
    cam_xf.AddTranslateOp().Set(CAMERA_LOCAL_OFFSET)
    print(f"Camera added at {CAMERA_PRIM_PATH}, local offset {CAMERA_LOCAL_OFFSET} -- verify mount visually")

    # --- Physics scene (skip if your stage already has one) ---
    if not stage.GetPrimAtPath("/World/PhysicsScene").IsValid():
        UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
        print("Added /World/PhysicsScene")

    # --- Light (matches your documented Week 6 fix) ---
    if not stage.GetPrimAtPath("/World/DistantLight").IsValid():
        light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        light.CreateIntensityAttr(LIGHT_INTENSITY)
        print(f"Added /World/DistantLight, intensity {LIGHT_INTENSITY}")

    # Clean up the field file's standalone-only sanity-check prims if present
    for p in ["/World/PhysicsScene_STANDALONE_ONLY", "/World/DistantLight_STANDALONE_ONLY"]:
        prim = stage.GetPrimAtPath(p)
        if prim.IsValid():
            stage.RemovePrim(p)
            print(f"Removed {p} (duplicate of the real one just added)")

    stage.GetRootLayer().Save()
    print("Saved. Now: Play once to confirm the robot doesn't sink, then rebuild the ActionGraph (see checklist).")


main()
