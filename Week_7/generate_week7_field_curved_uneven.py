#!/usr/bin/env python3
"""
generate_week7_field_curved_uneven.py

Builds ONE combined edge-case field for Week 7 Phase 0:
  - Curved rows (parallel rows following a shared lateral curve, like a
    contour-planted field)
  - Uneven row spacing (random gap between adjacent rows, not a fixed 1.8 m)
  - Uneven terrain (rolling height variation + micro-roughness under the
    whole field, with the ground mesh acting as both the visual ground and
    the physics collider)

Consistent with the established field convention (see week6-status.md):
  - Rows run primarily along Z, aisle (no-row corridor) centered on x=0
  - Row length ~14 m (Z from -7 to +7)
  - Robot start pose historically (0, 0.24, 6), driving toward -Z

Because terrain now has height variation, Y at the start pose is NOT 0.24
anymore -- it's terrain_height(0, 6) + 0.24. The script prints the exact
recommended start pose at the end. Update Carter's start transform in
Isaac Sim to that value.

THIS SCRIPT DOES NOT ADD: the Carter robot, camera, lights, ActionGraph,
or a PhysicsScene beyond a minimal one for standalone sanity-checking.
It only authors /World/Ground and /World/CropField. See the "How to use
this file" section printed at the end for the two ways to bring this
into your working stage.

Run with the Isaac Sim python venv (has usd-core + pillow already):
    source ~/env_isaacsim/bin/activate
    python3 generate_week7_field_curved_uneven.py
"""

import math
import random

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

# ---------------------------------------------------------------------------
# CONFIG -- tune these, everything below reads from here
# ---------------------------------------------------------------------------

OUTPUT_PATH = "week7_field_curved_uneven.usd"

SEED = 42  # fixed seed -> same field every run, needed for repeatable benchmarks

# Row layout
ROWS_PER_SIDE = 7  # -> 14 rows total, satisfies "10-15 rows"
ROW_LENGTH = 14.0  # along Z, matches existing convention: Z from -7 to +7
ROW_Z_MIN, ROW_Z_MAX = -ROW_LENGTH / 2.0, ROW_LENGTH / 2.0

AISLE_HALF_WIDTH = 0.9  # gap from x=0 to the first row on each side (nominal)
ROW_SPACING_MIN = 1.4  # matches the on-hold Week 6 uneven generator's range
ROW_SPACING_MAX = 2.2

# Curve (applied identically to every row's centerline -> rows stay parallel)
CURVE_AMPLITUDE = 0.9  # meters of lateral deviation at the curve's peak
CURVE_WAVELENGTH = 14.0  # one full sine cycle across the row length (gentle S-curve)
CURVE_PHASE_Z = 6.0  # z at which offset=0 -- MUST equal the robot's start z.
# Without this, the curve's own offset at the start pose (z=6) was 0.39 m --
# a perfectly centered robot would already read as 0.39 m off-track at tick
# one, before it has moved. Phase-locking the curve to the start z removes
# that artifact. If you change the robot's start z, change this to match.
# NOTE: at these defaults, peak dx/dz ~= 2*pi*0.9/14 ~= 0.40 (~22 deg). This is
# intentionally aggressive enough to break a lookahead-free corridor-follower,
# per the locked Week 6 pass/fail criteria ("expect corner-cutting/lag").
# Reduce CURVE_AMPLITUDE if the robot can't survive it at all during baseline.

# Terrain (rolling bumps + micro-roughness)
TERRAIN_BUMP_AMPLITUDE = 0.06  # meters, rolling furrow-scale variation
TERRAIN_BUMP_WAVELENGTH = 2.5  # meters
TERRAIN_NOISE_AMPLITUDE = 0.015  # meters, small per-vertex roughness
TERRAIN_GRID_CELL = 0.5  # meters, ground mesh resolution
TERRAIN_MARGIN = 2.0  # meters, extra ground beyond the outermost row

# Plants (proxy geometry per row)
PLANT_SPACING = 0.4  # meters along each row
PLANT_SPACING_JITTER = 0.08
PLANT_HEIGHT = 0.3
PLANT_RADIUS = 0.05
PLANT_COLOR = Gf.Vec3f(0.12, 0.55, 0.15)  # saturated green -- VERIFY against your
# calibrated HSV thresholds (val_min 40) once this loads in Isaac; you may need
# to redo the balance_bias calibration exactly like you did after the lighting fix.

# Ground physics material
GROUND_STATIC_FRICTION = 0.6
GROUND_DYNAMIC_FRICTION = 0.6
GROUND_RESTITUTION = 0.0
GROUND_COLOR = Gf.Vec3f(0.35, 0.25, 0.15)  # brown dirt

random.seed(SEED)


# ---------------------------------------------------------------------------
# Terrain + curve functions
# ---------------------------------------------------------------------------

def terrain_height(x: float, z: float) -> float:
    """Rolling bumps + light noise. Deterministic given (x, z) and SEED-derived
    phase offsets, so the same (x, z) always returns the same height even
    though we don't cache it."""
    rolling = TERRAIN_BUMP_AMPLITUDE * math.sin(2 * math.pi * x / TERRAIN_BUMP_WAVELENGTH) \
        * math.cos(2 * math.pi * z / (TERRAIN_BUMP_WAVELENGTH * 1.3))
    # deterministic pseudo-noise from coordinates (not random.random(), so it
    # doesn't depend on call order -- safe to call out of order / repeatedly)
    noise = TERRAIN_NOISE_AMPLITUDE * math.sin(12.9898 * x + 78.233 * z) * math.cos(4.1414 * x - 2.72 * z)
    return rolling + noise


def curve_offset(z: float) -> float:
    """Lateral (x) offset applied to every row at height z, and also the
    true corridor centerline's x position at that z (the aisle is nominally
    x=0, so it shifts by exactly this same amount). Phase-locked so offset=0
    at z=CURVE_PHASE_Z (the robot start z) -- keep this formula identical in
    any logger that scores deviation against the curve, not against x=0."""
    return CURVE_AMPLITUDE * math.sin(2 * math.pi * (z - CURVE_PHASE_Z) / CURVE_WAVELENGTH)


# ---------------------------------------------------------------------------
# Row layout: irregular spacing, symmetric-ish about the center aisle
# ---------------------------------------------------------------------------

def build_row_x_positions():
    """Returns a flat list of nominal (pre-curve) x positions, one per row,
    built outward from the aisle on each side with randomized gaps."""
    positions = []
    for side in (-1, 1):
        x = side * AISLE_HALF_WIDTH  # first row sits at the aisle edge, matching
        positions.append(x)          # the established ~0.9 m half-spacing convention
        for _ in range(ROWS_PER_SIDE - 1):
            gap = random.uniform(ROW_SPACING_MIN, ROW_SPACING_MAX)
            x = x + side * gap
            positions.append(x)
    return sorted(positions)


# ---------------------------------------------------------------------------
# USD authoring
# ---------------------------------------------------------------------------

def make_ground_mesh(stage, row_positions):
    x_min = min(row_positions) - TERRAIN_MARGIN
    x_max = max(row_positions) + TERRAIN_MARGIN
    z_min = ROW_Z_MIN - TERRAIN_MARGIN
    z_max = ROW_Z_MAX + TERRAIN_MARGIN

    nx = int(round((x_max - x_min) / TERRAIN_GRID_CELL)) + 1
    nz = int(round((z_max - z_min) / TERRAIN_GRID_CELL)) + 1

    points = []
    for iz in range(nz):
        z = z_min + iz * TERRAIN_GRID_CELL
        for ix in range(nx):
            x = x_min + ix * TERRAIN_GRID_CELL
            y = terrain_height(x, z)
            points.append(Gf.Vec3f(x, y, z))

    face_vertex_counts = []
    face_vertex_indices = []
    for iz in range(nz - 1):
        for ix in range(nx - 1):
            i0 = iz * nx + ix
            i1 = i0 + 1
            i2 = i0 + nx + 1
            i3 = i0 + nx
            face_vertex_counts.append(4)
            face_vertex_indices.extend([i0, i1, i2, i3])

    mesh_path = "/World/Ground/Terrain"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateDisplayColorAttr([GROUND_COLOR])
    mesh.CreateSubdivisionSchemeAttr("none")

    prim = mesh.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr().Set("none")  # static terrain: exact mesh, no approximation

    # Physics material (friction) bound for physics purpose
    mat_path = "/World/Ground/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    phys_mat_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys_mat_api.CreateStaticFrictionAttr().Set(GROUND_STATIC_FRICTION)
    phys_mat_api.CreateDynamicFrictionAttr().Set(GROUND_DYNAMIC_FRICTION)
    phys_mat_api.CreateRestitutionAttr().Set(GROUND_RESTITUTION)
    binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    binding_api.Bind(material, materialPurpose="physics")

    return x_min, x_max, z_min, z_max


def make_plant_material(stage):
    mat_path = "/World/CropField/PlantMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, mat_path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(PLANT_COLOR)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    shader_output = shader.ConnectableAPI().GetOutput("surface")
    if not shader_output:
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def make_rows(stage, row_positions, plant_material):
    crop_root = UsdGeom.Xform.Define(stage, "/World/CropField")
    plant_count = 0

    for row_idx, x_nominal in enumerate(row_positions):
        row_path = f"/World/CropField/Row_{row_idx:02d}"
        UsdGeom.Xform.Define(stage, row_path)

        z = ROW_Z_MIN
        while z <= ROW_Z_MAX:
            x = x_nominal + curve_offset(z)
            y_ground = terrain_height(x, z)

            plant_path = f"{row_path}/Plant_{plant_count:04d}"
            cyl = UsdGeom.Cylinder.Define(stage, plant_path)
            cyl.CreateRadiusAttr(PLANT_RADIUS)
            cyl.CreateHeightAttr(PLANT_HEIGHT)
            cyl.CreateAxisAttr("Y")
            xf = UsdGeom.Xformable(cyl)
            xf.AddTranslateOp().Set(Gf.Vec3d(x, y_ground + PLANT_HEIGHT / 2.0, z))
            UsdShade.MaterialBindingAPI.Apply(cyl.GetPrim()).Bind(plant_material)

            plant_count += 1
            z += PLANT_SPACING + random.uniform(-PLANT_SPACING_JITTER, PLANT_SPACING_JITTER)

    return plant_count


def make_minimal_standalone_extras(stage):
    """Only so the file is sanity-checkable on its own (e.g. open it solo and
    confirm it loads / looks right). If you reference this into your existing
    working stage, DELETE these two prims to avoid duplicating your real
    PhysicsScene and light."""
    from pxr import UsdLux
    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene_STANDALONE_ONLY")
    light = UsdLux.DistantLight.Define(stage, "/World/DistantLight_STANDALONE_ONLY")
    light.CreateIntensityAttr(3000.0)


def main():
    stage = Usd.Stage.CreateNew(OUTPUT_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())

    row_positions = build_row_x_positions()
    x_min, x_max, z_min, z_max = make_ground_mesh(stage, row_positions)
    plant_material = make_plant_material(stage)
    plant_count = make_rows(stage, row_positions, plant_material)
    make_minimal_standalone_extras(stage)

    stage.GetRootLayer().Save()

    start_x, start_z = 0.0, 6.0
    start_y = terrain_height(start_x, start_z) + 0.24

    print("=" * 70)
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Rows: {len(row_positions)} ({ROWS_PER_SIDE} per side)")
    print(f"Row x-positions (nominal, pre-curve): {[round(p, 2) for p in row_positions]}")
    print(f"Ground mesh extent: x [{x_min:.2f}, {x_max:.2f}]  z [{z_min:.2f}, {z_max:.2f}]")
    print(f"Plants placed: {plant_count}")
    print()
    print(f"RECOMMENDED ROBOT START POSE: ({start_x:.3f}, {start_y:.4f}, {start_z:.3f})")
    print("  (terrain is NOT flat here -- Y is no longer a fixed 0.24, it's")
    print("   terrain_height(0, 6) + 0.24. Set Carter's start transform to this.)")
    print()
    print("How to use this file:")
    print("  Option A (recommended): in your existing working stage, delete the")
    print("  old /World/CropField and /World/Ground (or whatever your Week 6")
    print("  field's ground/row prims are named), then add this file as a")
    print("  sublayer or reference so your Carter robot, camera, lights and")
    print("  ActionGraph are untouched. Also delete this file's")
    print("  *_STANDALONE_ONLY prims after referencing -- they'll duplicate")
    print("  your real PhysicsScene/light.")
    print("  Option B: open this file directly in Isaac Sim, then re-add Carter,")
    print("  the camera, and rebuild/copy the ActionGraph chains into it.")
    print("=" * 70)


if __name__ == "__main__":
    main()
