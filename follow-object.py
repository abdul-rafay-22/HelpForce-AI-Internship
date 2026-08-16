from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.objects import DynamicCuboid

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

franka = world.scene.add(
    Franka(prim_path="/World/Franka", name="franka", position=np.array([0.0, 0.0, 0.0]))
)

cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=np.array([0.5, 0.0, 0.025]),
        scale=np.array([0.05, 0.05, 0.05]),
        color=np.array([0.1, 0.8, 0.2])
    )
)

controller = RMPFlowController(name="rmpflow", robot_articulation=franka)
world.reset()

orientation = np.array([0.0, 1.0, 0.0, 0.0])
while app.is_running():
    world.step(render=True)
    if not world.is_playing():
        continue
    cube_pos, cube_rot = cube.get_world_pose()
    
    robot_target_pos = cube_pos + np.array([0.0, 0.0, 0.20])
    actions = controller.forward(
        target_end_effector_position=robot_target_pos,
        target_end_effector_orientation=orientation
    )
    franka.apply_action(actions)

app.close()
