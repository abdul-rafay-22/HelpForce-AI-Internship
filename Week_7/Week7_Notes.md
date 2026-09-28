# Week 7 — Edge-Case Field & Baseline Setup
**Intern:** Abdul Rafay
**Track:** Wheeled Robot Simulation — Agriculture Row-Following (Isaac Sim)
**Mentor:** Arslan Ansar · **Reporting to:** Usman Ali
**Date:** 29 September 2026

---

## Summary

Week 7 scope is curved rows, uneven row spacing/terrain, lighting, and
obstacle edge cases on top of the Week 6 straight-row baseline. This commit
covers the first piece: a combined curved + uneven-spacing + rough-terrain
test field, the robot/camera/scene setup to load it, and a benchmarking
logger built specifically for it. **The ActionGraph rebuild and the actual
baseline run are still pending** — they require Isaac Sim's GUI and haven't
been executed yet. Nothing below should be read as a passing result; it is
the environment and tooling the baseline run will use.

---

## What was completed

### 1. Combined edge-case field
`generate_week7_field_curved_uneven.py` builds one field combining three of
the four Week 7 edge cases in a single test:
- **Curved rows** — all rows follow a shared lateral sine curve (amplitude
  0.9 m, wavelength 14 m), so they stay parallel (contour-planting look)
  while breaking a lookahead-free corridor-follower's straight-line
  assumption, per the Week 6 pass/fail criteria.
- **Uneven row spacing** — adjacent-row gaps randomized 1.4–2.2 m (seeded,
  reproducible) instead of the fixed 1.8 m straight-row spacing.
- **Uneven terrain** — a rolling-bump + micro-roughness height field under
  the whole scene, authored as the same mesh that serves as the physics
  collider (friction material attached).

14 rows (7 per side), aisle nominally ±0.9 m, 495 plants, ~512 total prims.

### 2. Two field bugs found and fixed before any run was attempted
- **Start-pose curve offset was nonzero.** The curve's lateral offset at the
  robot's start z (6.0 m) worked out to +0.39 m — a perfectly centered robot
  would have read as already off-track at tick one. Fixed by phase-locking
  the curve (`CURVE_PHASE_Z = 6.0`) so offset is exactly 0 at the start z.
- **The straight-row deviation metric doesn't apply here.** The curve's peak
  lateral swing (0.9 m) exceeds the Week 6 fail threshold (0.6 m). A logger
  that measures deviation from a fixed x = 0 — correct for straight rows —
  would fail a perfectly-driven curved run regardless of controller quality.
  Fixed with a purpose-built logger (below) that scores against the moving
  curve centerline instead.

### 3. Robot/scene setup script
`setup_week7_robot_and_scene.py` (run inside Isaac Sim's Script Editor, not
a terminal — needs live Nucleus/stage APIs): references Carter v1 onto the
field, sets its start transform to the field's printed start pose
`(0.000, 0.2518, 6.000)`, adds a camera prim, and adds a PhysicsScene and
Distant Light matching the Week 6 working lighting fix (~3000 intensity).
Not yet run against the live scene — written and reviewed, not verified in
Isaac Sim.

### 4. Curved-field logger
`week7_curved_logger_node.py` — same parameter convention as Week 6's
`test_logger_node.py` (`run_length_m`, `fail_deviation_m`, `timeout_sec`,
`results_csv`) but scores deviation against the field's curve centerline
instead of a fixed x = 0. Note: its `CURVE_AMPLITUDE` / `CURVE_WAVELENGTH` /
`CURVE_PHASE_Z` constants are duplicated from the field generator (not
imported) — if the field's curve is retuned, this file must be updated to
match or its results are meaningless.

### 5. Full run guide written
A step-by-step guide covering field generation → Isaac Sim launch → scene
setup + visual verification → ActionGraph rebuild (exact port-to-port
wiring for the camera, odometry, and drive chains, carrying forward every
Week 6 hard-won fix: Dt wiring, linear/angular axis assignment, wheel
velocity limits) → per-chain ROS 2 verification → baseline run procedure →
reading results → troubleshooting. Kept outside this repo per its own
platform; available on request.

---

## Outstanding (blocking Phase 0 for Week 7)

- ActionGraph rebuild in Isaac Sim (camera, odometry, drive chains) —
  GUI work, not done from this environment.
- First baseline run on the curved+uneven field, logged via
  `week7_curved_logger_node.py`.
- Detection-rate capture (`/perception/row_detected` echo) during that run.
- Lighting edge case (likely a parameter sweep on existing Distant Light,
  no new field needed).
- Obstacle edge case — **not a tuning task**: no obstacle sensing exists in
  the current perception pipeline (HSV-green only). Needs new sensing
  (proximity/contact) and new control logic; scoped to a minimal
  stop-before-contact behavior for this week, not steer-around.
- Curved-row tuning is explicitly out of scope for the baseline — corner-
  cutting/lag is an expected, documented failure mode (no curvature
  lookahead in the current controller), not something to chase to zero.
- Before/after table, robustness report, benchmark matrix, technical
  write-up, and demo video — all downstream of the baseline run above.

## Files in this commit
- `generate_week7_field_curved_uneven.py` — combined curved/uneven-spacing/
  rough-terrain field generator (standalone `usd-core`, no Isaac Sim needed)
- `setup_week7_robot_and_scene.py` — robot/camera/light/physics setup,
  run inside Isaac Sim's Script Editor
- `week7_curved_logger_node.py` — deviation logger scored against the
  curve centerline, for use only with the field above
