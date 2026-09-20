#!/usr/bin/env python3
"""
Week 6 Field Generator — Agriculture Row-Following (Isaac Sim)

Extends generate_week5_field.py: 8 rows (was 6), much denser in-row plant
spacing, 3 discrete plant SIZE tiers (new growth / under-growth / mature),
and a 5th HEALTH palette for dead/dying plants. Same architecture as Week
5 on purpose — one PointInstancer per row, ground-only collision, single
shared leaf texture tinted per palette — because that's the part of Week 5
that was already right and there's no reason to pay the perf cost twice.

WHAT CHANGED FROM WEEK 5 AND WHY
---------------------------------
- NUM_ROWS: 6 -> 8 (your ask: "make it till 7-8 rows")
- ROW_SPACING: left at 1.8 m, UNCHANGED. Week 5's own comment says your
  Week 4 HSV/column-histogram calibration was tuned against this row
  spacing — widening the field by adding rows, not by tightening row
  gap, was the whole point of splitting "dense" into two separate knobs.
  If you *do* want to test narrower row spacing at some point, that's a
  deliberate edge case for Phase 2, not something to fold in here
  silently.
- PLANT_SPACING: 0.9 -> 0.25 m (within-row spacing). This is the number
  that was prepped-but-never-loaded at the end of your Week 5 session —
  using it now instead of inventing a new value.
- SIZE now has 3 discrete tiers instead of one continuous JITTER_SCALE
  band, applied as a scale multiplier on the same base prototype
  geometry (no new prototypes needed for size — cheap).
- HEALTH now has 5 palettes instead of 4: the Week 5 palettes
  (Mature / Young / Flowering / Stressed) plus a new Dead palette
  (browner, droopy leaf angle, fewer leaves, no bloom).
- Size and health are drawn independently per plant, with a *soft* bias
  so small plants skew toward the Young palette and tall plants skew
  toward Mature — matching your framing (small = new growth, tall =
  fully grown) without hard-coding it (a small dead plant, a tall
  stressed plant etc. can still occur, which is realistic).
- Each row's PointInstancer also gets two extra primvars — "health" and
  "sizeTier" (token arrays, one entry per plant, same order as
  positions) — not required for rendering, but free ground-truth labels
  per plant instance if you ever want to score detection against known
  plant condition/size instead of just row position. Ignore them if you
  don't need them.

KNOWN INTERACTION WITH YOUR DETECTOR — READ THIS BEFORE PHASE 3
------------------------------------------------------------------
Your Week 3-5 notes record green vegetation at HSV hue 47-61 vs soil at
0-25. The Dead palette's leaf color sits in the hue gap between those
(~25-40, browner than Stressed's 0.45/0.42/0.14 olive-yellow). That is
NOT a bug — it's the Week 6 checklist's "detection accuracy" edge case
made concrete: dead plants are a real, expected failure mode for a pure
green-hue detector, not a rendering mistake to fix by re-tinting them
greener. Log it as a Phase 3 baseline failure, then decide in Phase 4
whether it's in-scope to fix (e.g. widen the accepted hue band) or an
accepted limitation you note in the report.

Requires: pip install usd-core pillow
Run:      python3 generate_week6_field.py
Output:   week6_field.usd  +  textures/leaf_diffuse.png (same generation
           approach as Week 5 — one procedural texture, reused by every
           palette via per-material tint, so this doesn't multiply your
           texture memory cost by 5)

Keep textures/ next to week6_field.usd when you move it — same absolute-
path caveat as Week 5: this is binary crate-format USD, so if you move
the file to another machine/folder, re-run the script from the new
location to rebake the texture path rather than hand-editing it.

Import into Isaac Sim the same way as before — reference/drag-drop into
your stage, don't double-click-open it over your existing scene.
"""

import math
import os
import random

from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Gf, Sdf
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# CONFIG — tune here
# ---------------------------------------------------------------------------
OUTPUT_PATH = "week6_field.usd"

NUM_ROWS = 28            # was 12 — ground size is *always* derived directly
                          # from NUM_ROWS and FIELD_MARGIN (see field_width
                          # in main()), so the field never has rows falling
                          # short of the ground plane by definition. The
                          # blank corners in your screenshots are the
                          # camera's viewing angle on a square-ish plane,
                          # not missing crop coverage — more rows still
                          # makes the field itself bigger, which is what
                          # you're actually asking for here.
ROW_SPACING = 1.8        # m — UNCHANGED since Week 3/4. Your perception
                          # calibration was tuned against this. Don't touch
                          # it here; narrower row spacing is a Phase 2
                          # edge-case variant, not this baseline field.
PLANT_SPACING = 0.25     # m — was 0.9 in Week 5. This is the spacing that
                          # was prepped but never loaded at the end of
                          # Week 5 — used now for real.
ROW_LENGTH = 14.0        # m — unchanged from Week 5
FIELD_MARGIN = 0.3       # m of bare ground around the crop rows — was 0.6,
                          # tightened further so the ground plane sits right
                          # at the edge of the crop rows.

JITTER_XY = 0.08         # m, per-plant lateral/longitudinal jitter
JITTER_ROT_DEG = 15.0    # degrees, random yaw per plant
SCALE_JITTER = 0.15      # +/- fractional jitter applied on top of each
                          # size tier's base scale range, so plants within
                          # the same tier still aren't identical

FURROW_WIDTH = 0.25
FURROW_HEIGHT = 0.04
FURROW_COLOR = Gf.Vec3f(0.38, 0.27, 0.16)


SOIL_BASE_COLORS = [
    Gf.Vec3f(0.55, 0.42, 0.25),
    Gf.Vec3f(0.48, 0.36, 0.20),
    Gf.Vec3f(0.60, 0.47, 0.30),
]
SOIL_TEXTURE_NAME = "soil_diffuse.png"
SOIL_TEXTURE_REPEATS_PER_METER = 0.4  # texture tiles roughly every 2.5 m

# ---------------------------------------------------------------------------
# HEALTH palettes — what color/leaf-condition a plant is.
# Colors pulled toward the reference vegetation images: teal-tinted new
# growth, richer forest green for mature, purple-flower accent instead of
# pale cream, autumn-tan stressed, and a clearly-legible maroon-red dead
# (was a muddy brown-yellow — see note in the module docstring area above
# about what this trades away for the detector).
# ---------------------------------------------------------------------------
PLANT_PALETTES = [
    {"name": "CropA_Mature", "leaf": Gf.Vec3f(0.10, 0.38, 0.12), "stem": Gf.Vec3f(0.22, 0.32, 0.10)},
    {"name": "CropB_Young", "leaf": Gf.Vec3f(0.16, 0.55, 0.42), "stem": Gf.Vec3f(0.25, 0.45, 0.30)},
    {"name": "CropC_Flowering", "leaf": Gf.Vec3f(0.14, 0.36, 0.14), "stem": Gf.Vec3f(0.20, 0.30, 0.11),
     "accent": Gf.Vec3f(0.55, 0.20, 0.45)},
    {"name": "CropD_Stressed", "leaf": Gf.Vec3f(0.58, 0.42, 0.12), "stem": Gf.Vec3f(0.40, 0.30, 0.10)},
    # Dead/dying — maroon-red now, not brown-yellow. Easy to eyeball as
    # unhealthy at a glance; also easier for the color detector to
    # separate from soil than the old ambiguous-hue version was.
    {"name": "CropE_Dead", "leaf": Gf.Vec3f(0.45, 0.14, 0.10), "stem": Gf.Vec3f(0.30, 0.16, 0.08),
     "droop": True},
]
PALETTE_BY_NAME = {p["name"]: p for p in PLANT_PALETTES}

# ---------------------------------------------------------------------------
# SIZE tiers — new growth / under-growth / mature, applied as a scale
# multiplier on the base prototype. Measured directly off the generated
# geometry (BBoxCache), not guessed: the layered-bush prototype is ~0.49-
# 0.51 m tall unscaled — taller than the old single-ring version, so these
# ranges were recalculated to land back on the same real-world target
# heights as before (small ~0.25-0.32 m, medium ~0.34-0.44 m,
# tall ~0.59-0.79 m).
# ---------------------------------------------------------------------------
SIZE_TIERS = [
    {"name": "small_new", "scale_range": (0.36, 0.62)},        # ~0.18-0.31 m
    {"name": "medium_growing", "scale_range": (0.60, 1.05)},   # ~0.30-0.53 m
    {"name": "tall_mature", "scale_range": (1.05, 1.95)},      # ~0.53-0.98 m
]

# Per-tier health distribution: soft bias only. Small plants skew Young,
# tall plants skew Mature; Dead is held roughly constant across tiers
# (dying isn't exclusive to any one growth stage). Each dict must sum to 1.0.
TIER_HEALTH_BIAS = {
    "small_new": {
        "CropA_Mature": 0.10, "CropB_Young": 0.48, "CropC_Flowering": 0.15,
        "CropD_Stressed": 0.15, "CropE_Dead": 0.12,
    },
    "medium_growing": {
        "CropA_Mature": 0.22, "CropB_Young": 0.22, "CropC_Flowering": 0.24,
        "CropD_Stressed": 0.20, "CropE_Dead": 0.12,
    },
    "tall_mature": {
        "CropA_Mature": 0.45, "CropB_Young": 0.05, "CropC_Flowering": 0.25,
        "CropD_Stressed": 0.13, "CropE_Dead": 0.12,
    },
}
# Tier itself is drawn uniformly-ish: fewer tall plants than small/medium
# reads more like a real field (most plants are still growing at any
# snapshot in time), but this is a cosmetic choice, not a constraint —
# change freely.
SIZE_TIER_WEIGHTS = {"small_new": 0.30, "medium_growing": 0.42, "tall_mature": 0.28}

TEXTURE_DIR = "textures"
LEAF_TEXTURE_NAME = "leaf_diffuse.png"
LEAF_TEXTURE_B_NAME = "leaf_diffuse_b.png"  # second shape variant, see build_plant_prototype
LEAF_TEXTURE_BASE_COLOR = (90, 150, 60)

random.seed(42)  # reproducible layout — change the seed for a fresh field


def weighted_choice(weight_dict):
    names = list(weight_dict.keys())
    weights = list(weight_dict.values())
    return random.choices(names, weights=weights, k=1)[0]


def generate_leaf_texture(path, size=256, pinch=0.35, aspect=1.0):
    """Procedural leaf-shaped alpha-cutout texture. `pinch` controls how
    pointed the tip is (higher = more pointed), `aspect` stretches width
    vs height (used to make a second, visually distinct leaf shape so
    every plant on the field isn't built from identical leaf silhouettes
    — was 1 shared texture for all 5 palettes, now 2)."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size / 2, size / 2
    rx, ry = size * 0.42 * aspect, size * 0.46

    pts = []
    for i in range(64):
        t = i / 64 * 2 * math.pi
        p = 1.0 - pinch * abs(math.cos(t)) ** 3
        pts.append((cx + rx * p * math.cos(t), cy + ry * p * math.sin(t)))
    draw.polygon(pts, fill=LEAF_TEXTURE_BASE_COLOR + (255,))

    vein_color = (60, 110, 40, 200)
    draw.line([(cx, cy - ry * 0.9), (cx, cy + ry * 0.9)], fill=vein_color, width=3)
    for k in range(-3, 4):
        if k == 0:
            continue
        y0 = cy + k * ry * 0.22
        x1 = cx + rx * 0.6 * (1 if k % 2 == 0 else -1)
        draw.line([(cx, y0), (x1, y0 + ry * 0.12)], fill=vein_color, width=2)

    px = img.load()
    for _ in range(2000):
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        r, g, b, a = px[x, y]
        if a > 0:
            j = random.randint(-15, 15)
            px[x, y] = (max(0, min(255, r + j)), max(0, min(255, g + j)), max(0, min(255, b + j)), a)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)
    return path


def generate_soil_texture(path, size=512):
    """Tileable dirt texture: low-frequency color patches (blend of the 3
    soil base tones) for the 'damp patch here, dry patch there' look,
    fine per-pixel grain on top, and scattered darker clump/pebble
    speckles. Replaces the old flat per-face vertex-color ground, which
    read as a handful of solid-color tiles rather than dirt."""
    # Low-res color field, upscaled with smoothing -> soft patches instead
    # of hard tile edges.
    low = 24
    patch = Image.new("RGB", (low, low))
    for y in range(low):
        for x in range(low):
            c = random.choice(SOIL_BASE_COLORS)
            jitter = random.uniform(-0.04, 0.04)
            patch.putpixel((x, y), (
                max(0, min(255, int((c[0] + jitter) * 255))),
                max(0, min(255, int((c[1] + jitter) * 255))),
                max(0, min(255, int((c[2] + jitter) * 255))),
            ))
    img = patch.resize((size, size), Image.BICUBIC)

    # Fine grain, same technique as the leaf texture.
    px = img.load()
    for y in range(size):
        for x in range(size):
            r, g, b = px[x, y]
            j = random.randint(-10, 10)
            px[x, y] = (max(0, min(255, r + j)), max(0, min(255, g + j)), max(0, min(255, b + j)))

    # Darker clumps/pebbles scattered on top.
    draw = ImageDraw.Draw(img)
    for _ in range(int(size * size * 0.0015)):
        cx, cy = random.randint(0, size - 1), random.randint(0, size - 1)
        r = random.randint(1, 4)
        darken = random.uniform(0.5, 0.75)
        base = px[cx, cy]
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            fill=(int(base[0] * darken), int(base[1] * darken), int(base[2] * darken)),
        )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.save(path)
    return path


def make_soil_material(stage, path, texture_abs_path):
    mat = UsdShade.Material.Define(stage, path)

    st_reader = UsdShade.Shader.Define(stage, path + "/StReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex = UsdShade.Shader.Define(stage, path + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_abs_path)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
    rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    surf = UsdShade.Shader.Define(stage, path + "/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    surf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat


def make_material(stage, path, color, roughness=0.85):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def build_ground(stage, width, length, texture_abs_path):
    path = "/World/Ground"
    mesh = UsdGeom.Mesh.Define(stage, path)

    subdiv = 8  # mesh density unchanged — detail now comes from the texture,
                 # not per-face vertex color, so this doesn't need to grow
    step_x = width / subdiv
    step_y = length / subdiv

    pts, faces, counts, st_coords = [], [], [], []
    for j in range(subdiv + 1):
        for i in range(subdiv + 1):
            pts.append(Gf.Vec3f(-width / 2 + i * step_x, 0.0, -length / 2 + j * step_y))
            st_coords.append(Gf.Vec2f(
                i * step_x * SOIL_TEXTURE_REPEATS_PER_METER,
                j * step_y * SOIL_TEXTURE_REPEATS_PER_METER,
            ))
    for j in range(subdiv):
        for i in range(subdiv):
            a = j * (subdiv + 1) + i
            b = a + 1
            c = a + (subdiv + 1) + 1
            d = a + (subdiv + 1)
            faces += [a, b, c, d]
            counts.append(4)

    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(faces)
    mesh.CreateExtentAttr([Gf.Vec3f(-width / 2, 0, -length / 2), Gf.Vec3f(width / 2, 0, length / 2)])

    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex")
    st.Set(st_coords)  # wrapS/T=repeat (set on the material) tiles this

    soil_mat = make_soil_material(stage, "/World/Materials/Soil", texture_abs_path)
    UsdShade.MaterialBindingAPI(mesh).Bind(soil_mat)

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    return mesh


def build_furrows(stage, num_rows, row_spacing, row_length):
    furrow_mat = make_material(stage, "/World/Materials/Furrow", FURROW_COLOR)
    start_x = -((num_rows - 1) * row_spacing) / 2
    for r in range(num_rows):
        x = start_x + r * row_spacing
        box = UsdGeom.Cube.Define(stage, f"/World/Furrows/Furrow_{r}")
        box.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(box)
        xf.AddScaleOp().Set(Gf.Vec3f(FURROW_WIDTH, FURROW_HEIGHT, row_length))
        xf.AddTranslateOp().Set(Gf.Vec3d(x, FURROW_HEIGHT / 2, 0.0))
        UsdShade.MaterialBindingAPI(box).Bind(furrow_mat)


def make_leaf_card_material(stage, path, texture_abs_path, leaf_color):
    mat = UsdShade.Material.Define(stage, path)

    st_reader = UsdShade.Shader.Define(stage, path + "/StReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_out = st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    base = Gf.Vec3f(*[c / 255.0 for c in LEAF_TEXTURE_BASE_COLOR])
    tint = Gf.Vec4f(leaf_color[0] / base[0], leaf_color[1] / base[1], leaf_color[2] / base[2], 1.0)

    tex = UsdShade.Shader.Define(stage, path + "/Texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_abs_path)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
    tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(tint)
    tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    rgb_out = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    a_out = tex.CreateOutput("a", Sdf.ValueTypeNames.Float)

    surf = UsdShade.Shader.Define(stage, path + "/Surface")
    surf.CreateIdAttr("UsdPreviewSurface")
    surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb_out)
    surf.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(a_out)
    surf.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.3)
    surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.62)  # was 0.75 — a
    # little more sheen reads less flat/matte-plastic, closer to real leaf surface
    mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
    return mat


def add_leaf_card(stage, path, width=0.09, height=0.13):
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([
        Gf.Vec3f(-width, 0, 0), Gf.Vec3f(width, 0, 0),
        Gf.Vec3f(width, height * 2, 0), Gf.Vec3f(-width, height * 2, 0),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateExtentAttr([Gf.Vec3f(-width, 0, 0), Gf.Vec3f(width, height * 2, 0)])
    mesh.CreateDoubleSidedAttr(True)
    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying")
    st.Set([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)])
    return mesh


def build_plant_prototype(stage, path, palette, texture_paths):
    """Bushy shrub shape: multiple stacked layers of leaf cards (was a
    single ring near the base), each layer narrower and more upright than
    the one below it, so the silhouette tapers like an actual shrub
    instead of reading as one ring of leaves on a stick. Leaves within a
    layer are rotation-staggered from the layer below so they don't line
    up radially.

    texture_paths is (texture_a, texture_b) — two differently-shaped leaf
    textures (see generate_leaf_texture). Each leaf card randomly binds
    to one or the other, so a single plant isn't built from one
    identical leaf silhouette repeated 15+ times.

    This still costs nothing extra per plant instance — there are only 5
    prototypes total (one per health palette), each referenced ~90 times
    by the PointInstancer. More leaf cards / an extra texture per
    prototype means more triangles and one more shared file, not 5x more
    meshes or textures.

    Dead plants (palette['droop']) get fewer layers, fewer leaves per
    layer, and a downward droop instead of the outward/upward splay —
    same "identifiable at a glance" logic as before, just applied per
    layer now."""
    proto = UsdGeom.Xform.Define(stage, path)

    stem_mat = make_material(stage, path + "/StemMat", palette["stem"])
    texture_a, texture_b = texture_paths
    leaf_mat_a = make_leaf_card_material(stage, path + "/LeafMatA", texture_a, palette["leaf"])
    leaf_mat_b = make_leaf_card_material(stage, path + "/LeafMatB", texture_b, palette["leaf"])

    crown = UsdGeom.Cylinder.Define(stage, path + "/Crown")
    crown.CreateHeightAttr(0.10)
    crown.CreateRadiusAttr(0.025)
    UsdGeom.Xformable(crown).AddTranslateOp().Set(Gf.Vec3d(0, 0.05, 0))
    UsdShade.MaterialBindingAPI(crown).Bind(stem_mat)

    is_dead = palette.get("droop", False)
    num_layers = 2 if is_dead else 4
    base_leaves_per_layer = 3 if is_dead else 5
    canopy_height = 0.16 if is_dead else 0.30  # local, before tier scaling

    for layer in range(num_layers):
        layer_t = (layer + 0.5) / num_layers  # 0 (bottom) .. ~1 (top)
        # Fewer leaves per layer as you go up (healthy only) -> tapered top.
        leaves_in_layer = max(2, base_leaves_per_layer - (0 if is_dead else layer))
        radius_base = 0.07 * (1.0 - 0.18 * layer)
        height_base = 0.06 + layer_t * canopy_height
        card_w = max(0.03, 0.085 - layer * 0.008)
        card_h = max(0.05, 0.12 - layer * 0.010)
        # Rotation offset per layer so leaves don't stack directly above
        # the ones in the layer below.
        layer_rot_offset = layer * 22.0

        for k in range(leaves_in_layer):
            angle_deg = k * (360.0 / leaves_in_layer) + layer_rot_offset + random.uniform(-12, 12)
            angle_rad = math.radians(angle_deg)
            height = height_base + random.uniform(-0.015, 0.015)
            out_radius = max(0.015, radius_base + random.uniform(-0.01, 0.01))
            if is_dead:
                tilt_deg = random.uniform(-60, -25)
            else:
                # Bottom layers splay outward more (~50 deg), top layers
                # stand more upright (~15 deg) -> rounded bush silhouette.
                tilt_deg = random.uniform(35, 55) - layer * 9

            card = add_leaf_card(stage, f"{path}/Leaf_{layer}_{k}", width=card_w, height=card_h)
            xf = UsdGeom.Xformable(card)
            xf.AddRotateZOp().Set(tilt_deg)
            xf.AddRotateYOp().Set(angle_deg)
            xf.AddTranslateOp().Set(Gf.Vec3d(
                out_radius * math.cos(angle_rad), height, out_radius * math.sin(angle_rad)
            ))
            UsdShade.MaterialBindingAPI(card).Bind(leaf_mat_a if random.random() < 0.5 else leaf_mat_b)

    if "accent" in palette:
        accent_mat = make_material(stage, path + "/AccentMat", palette["accent"])
        bloom = UsdGeom.Sphere.Define(stage, path + "/Bloom")
        bloom.CreateRadiusAttr(0.035)
        UsdGeom.Xformable(bloom).AddTranslateOp().Set(Gf.Vec3d(0, 0.06 + canopy_height + 0.03, 0))
        UsdShade.MaterialBindingAPI(bloom).Bind(accent_mat)

    return proto.GetPrim()


def build_crop_rows(stage, num_rows, row_spacing, plant_spacing, row_length, prototypes):
    """One PointInstancer per row, same as Week 5 — this is still the
    load-bearing perf decision at 0.25 m spacing / 8 rows (~450 plants
    total). Now also assigns each instance a size tier + health palette
    (with soft tier->health bias) instead of one continuous scale jitter,
    and records both as extra primvars for later ground-truth use."""
    start_x = -((num_rows - 1) * row_spacing) / 2
    plants_per_row = int(row_length / plant_spacing)
    proto_names = [p["name"] for p in PLANT_PALETTES]
    proto_index_by_name = {name: i for i, name in enumerate(proto_names)}

    tier_counts = {t["name"]: 0 for t in SIZE_TIERS}
    health_counts = {name: 0 for name in proto_names}

    for r in range(num_rows):
        row_x = start_x + r * row_spacing
        instancer = UsdGeom.PointInstancer.Define(stage, f"/World/CropRows/Row_{r}")
        instancer.CreatePrototypesRel().SetTargets([p.GetPath() for p in prototypes])

        positions, orientations, scales, proto_indices = [], [], [], []
        health_labels, size_labels = [], []

        for p_idx in range(plants_per_row):
            y = -row_length / 2 + p_idx * plant_spacing
            jx = random.uniform(-JITTER_XY, JITTER_XY)
            jy = random.uniform(-JITTER_XY, JITTER_XY)
            positions.append(Gf.Vec3f(row_x + jx, 0.0, y + jy))

            yaw_deg = random.uniform(-JITTER_ROT_DEG, JITTER_ROT_DEG)
            half = math.radians(yaw_deg) / 2.0
            orientations.append(Gf.Quath(math.cos(half), 0.0, math.sin(half), 0.0))

            tier_name = weighted_choice(SIZE_TIER_WEIGHTS)
            tier = next(t for t in SIZE_TIERS if t["name"] == tier_name)
            base_scale = random.uniform(*tier["scale_range"])
            s = base_scale * random.uniform(1 - SCALE_JITTER, 1 + SCALE_JITTER)
            # Non-uniform per-plant scale: independent jitter on height vs
            # width so plants aren't uniformly-proportioned clones at
            # different sizes — some slightly leggier, some slightly
            # squatter/bushier, like real plant-to-plant variation.
            s_y = s * random.uniform(0.85, 1.30)
            s_xz = s * random.uniform(0.80, 1.15)
            scales.append(Gf.Vec3f(s_xz, s_y, s_xz))

            health_name = weighted_choice(TIER_HEALTH_BIAS[tier_name])
            proto_indices.append(proto_index_by_name[health_name])

            health_labels.append(health_name)
            size_labels.append(tier_name)
            tier_counts[tier_name] += 1
            health_counts[health_name] += 1

        instancer.CreatePositionsAttr(positions)
        instancer.CreateOrientationsAttr(orientations)
        instancer.CreateScalesAttr(scales)
        instancer.CreateProtoIndicesAttr(proto_indices)

        # Extra ground-truth labels — plain custom attributes, NOT primvars.
        # Primvars get pushed into Hydra's shading pipeline; RTX doesn't
        # support token-array primvars and spams "Unsupported data type"
        # warnings for every leaf/crown prim under the instancer (this bit
        # us on the first render — fixed). Custom attributes just sit on
        # the prim as data — read them back with
        # prim.GetAttribute("health").Get() from a script, never touched
        # by the renderer.
        instancer.GetPrim().CreateAttribute(
            "health", Sdf.ValueTypeNames.TokenArray, custom=True).Set(health_labels)
        instancer.GetPrim().CreateAttribute(
            "sizeTier", Sdf.ValueTypeNames.TokenArray, custom=True).Set(size_labels)

    return plants_per_row, tier_counts, health_counts


def main():
    texture_a_path = generate_leaf_texture(
        os.path.join(TEXTURE_DIR, LEAF_TEXTURE_NAME), pinch=0.35, aspect=1.0)
    texture_a_abs = os.path.abspath(texture_a_path)
    texture_b_path = generate_leaf_texture(
        os.path.join(TEXTURE_DIR, LEAF_TEXTURE_B_NAME), pinch=0.12, aspect=1.15)
    texture_b_abs = os.path.abspath(texture_b_path)
    soil_texture_path = generate_soil_texture(os.path.join(TEXTURE_DIR, SOIL_TEXTURE_NAME))
    soil_texture_abs_path = os.path.abspath(soil_texture_path)

    stage = Usd.Stage.CreateNew(OUTPUT_PATH)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # WAS MISSING. Without this, USD
    # defaults an unset stage to centimeters — every number in this script
    # (1.8m row spacing, plant heights, etc.) was written assuming meters,
    # so leaving this unset was silently wrong the whole time. It didn't
    # show up until you composed the field against Carter v1, which DOES
    # correctly declare its own units (centimeters) — Isaac's auto unit-
    # conversion between the two produced that backwards 100x scale on
    # Carter instead of the expected 0.01x. This line fixes it at the
    # source instead of hand-editing Carter's Scale:unitsResolve, which
    # would just get recomputed wrong again on the next reload.
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

    field_width = (NUM_ROWS - 1) * ROW_SPACING + 2 * FIELD_MARGIN
    field_length = ROW_LENGTH + 2 * FIELD_MARGIN

    build_ground(stage, field_width, field_length, soil_texture_abs_path)
    build_furrows(stage, NUM_ROWS, ROW_SPACING, ROW_LENGTH)

    prototypes = [
        build_plant_prototype(stage, f"/World/Prototypes/{pal['name']}", pal, (texture_a_abs, texture_b_abs))
        for pal in PLANT_PALETTES
    ]

    plants_per_row, tier_counts, health_counts = build_crop_rows(
        stage, NUM_ROWS, ROW_SPACING, PLANT_SPACING, ROW_LENGTH, prototypes
    )

    stage.GetRootLayer().Save()

    total_plants = NUM_ROWS * plants_per_row
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  Rows: {NUM_ROWS} @ {ROW_SPACING} m spacing (unchanged from Week 3-5)")
    print(f"  Plants/row: {plants_per_row} @ {PLANT_SPACING} m spacing (was 0.9 m in Week 5)")
    print(f"  Total instanced plants: {total_plants}")
    print(f"  Size tier distribution: {tier_counts}")
    print(f"  Health distribution: {health_counts}")
    print("  Collision: ground only — no per-plant colliders")
    print(f"  Leaf textures: {texture_a_abs}, {texture_b_abs} (2 files, referenced by absolute path)")
    print("  NOTE: binary crate-format USD — if you move week6_field.usd to a")
    print("  different machine/folder, re-run this script from the new location")
    print("  to rebake the texture path.")
    print("  NOTE: CropE_Dead is now maroon-red (visually distinct), not the")
    print("  brown-yellow used before — that earlier color sat in the hue gap")
    print("  between soil and green vegetation on purpose, as a hard case for")
    print("  the detector. This version is easier to eyeball as dead but also")
    print("  easier for the detector to separate from soil, so it no longer")
    print("  stresses that particular edge case the same way.")


if __name__ == "__main__":
    main()
