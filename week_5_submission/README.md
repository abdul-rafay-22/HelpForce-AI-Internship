# Week 5 — Control & Autonomous Row-Following

This is my Week 5 work for the Ground Simulation programme: getting the robot to
drive itself down a crop row using the camera-based row detection from Week 4,
closed-loop, in Isaac Sim.

## Where this actually got to (honest version)

The core of the task works. Perception runs live, the whole ROS 2 → Isaac drive
chain is wired and functioning, and the robot did follow a row on its own before
drifting into it. What's *not* finished yet is the controller tuning and a proper
set of logged test runs — I ran out of a clean detection signal to tune against,
for reasons I explain below, and I'd rather hand this over being straight about
that than pretend it's polished.

So: system built and demonstrated, tuning and logging carried into next week.

## What's here

**`control/`** — the closed-loop control side:

- `perception_bridge_node.py` — turns the Week 4 detection (HSV mask + column
  histogram) into a live ROS 2 node. Publishes the row offset and a
  "did I actually see a row this frame" flag. Now also tracks the lane across
  frames instead of re-picking a row every frame (this was the Week 5 temporal
  improvement).
  **Note:** the copy in this repo is the earlier version — replace it with the
  temporal-tracking version from my working machine before relying on it.
- `row_follower_controller.py` — the PID (really PD) controller. Reads the offset,
  publishes `/cmd_vel`. Has a safety stop if the row is lost for too long, and
  smooths the offset before steering on it.
- `test_logger_node.py` — logs one closed-loop run: deviation from the row centre
  over time, plus a pass/fail result, into a CSV. Exits on its own when the run
  ends.
- `analyze_results.py` — turns the run CSVs into the success-rate and deviation
  numbers for the write-up.
- `perception_debug_snapshot.py` — grabs one camera frame, runs detection, and
  saves annotated images to disk so you can see what it's locking onto. Built this
  to debug the detection when the ROS debug-image topic wouldn't cooperate.
- `build_drive_graph.py` — builds the whole Action Graph drive chain in one run
  from Isaac's Script Editor, instead of wiring nodes by hand. (Wiring by hand kept
  crashing my machine, so this exists as the reliable path.)
- `isaac_recovery.py` — resets the robot + camera to known-good values after a
  crash. Isaac has no autosave and my machine crashed a lot, so this saved time.
- `run_week5_pipeline.sh` — starts perception + controller together.
- `WEEK5_GUIDE.md` — the full step-by-step: importing the field, the six checklist
  phases, the exact Action Graph node connections, tuning, and the tech-notes
  template.

**`field/`** — the simulation environment:

- `generate_week5_field.py` — procedurally builds the crop field (rows, furrows,
  soil colour, textured plants) and a leaf texture, as a `.usd`. Denser plant
  spacing is in here now. Run it locally before use — the leaf texture path is
  baked in at generation time, so it has to be generated on the machine it runs on.

## How to run it

You need ROS 2 Jazzy and Isaac Sim. Rough sequence (the guide has the full detail):

1. Generate the field: `cd field && python3 generate_week5_field.py`
2. In Isaac: load your stage, import the field, put the robot in a lane pointed
   down the row, pitch the camera down so it sees the ground.
3. Build the drive graph: paste `build_drive_graph.py` into Isaac's Script Editor
   and run it (or wire the nodes by hand per the guide).
4. Press Play.
5. Two terminals (both `source /opt/ros/jazzy/setup.bash` first):
   - `python3 control/perception_bridge_node.py`
   - `python3 control/row_follower_controller.py`
6. The robot follows the row.

## Things that bit me (so they don't bite the next person)

- **cv_bridge segfaults with NumPy 2.x on Jazzy.** Fix: `pip install "numpy<2"
  --break-system-packages`.
- **`cv2_to_imgmsg(..., encoding='bgr8')` throws `KeyError: 16`** on this build.
  The ROS debug-image topic is unreliable because of it — use the disk-saving
  debug snapshot instead.
- **The Differential Controller node has no reliable Exec Out** here — drive the
  Articulation Controller's Exec In straight off the playback tick.
- **Subscribe Twist outputs 3-vectors; the Differential Controller wants scalars.**
  Route through Break 3-Vector nodes (linear → X, angular → Z), or it won't connect.
- **Save Isaac constantly.** No autosave. I lost graph work to crashes more than
  once before this sank in.

## Known limitation

Row detection gets jumpy when the plants are sparse — the column histogram sees a
different pattern frame to frame, so the offset wobbles and the controller can't
tune against it cleanly. The denser field is the fix for that, and finishing the
tuning + logged runs on top of a stable signal is the first job next week.
