from isaacsim import SimulationApp
app = SimulationApp({"headless": False})

import numpy as np
from omni.isaac.core import World
from omni.isaac.franka import Franka
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.utils.types import ArticulationAction

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

franka = world.scene.add(
    Franka(prim_path="/World/Franka", name="franka", position=np.array([0.0, 0.0, 0.0]))
)

# FIXED: Smaller scale and explicit light mass to guarantee it doesn't slip
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Cube",
        name="cube",
        position=np.array([0.4, 0.0, 0.02]),
        scale=np.array([0.04, 0.04, 0.04]),
        color=np.array([0.9, 0.1, 0.1]),
        mass=0.01 
    )
)

controller = RMPFlowController(name="rmpflow", robot_articulation=franka)
world.reset()

orientation = np.array([0.0, 1.0, 0.0, 0.0])
open_grip = ArticulationAction(joint_positions=np.array([0.04, 0.04]))
close_grip = ArticulationAction(joint_positions=np.array([0.0, 0.0])) # Fully clamped

frames = 0
print("Running Physics-Fixed Pick and Place...")

while app.is_running():
    world.step(render=True)
    if not world.is_playing():
        continue
        
    frames += 1
    
    # 1. Hover and explicitly OPEN fingers wide so it doesn't knock the cube away
    if frames < 150:
        target = np.array([0.4, 0.0, 0.2]) 
        franka.gripper.apply_action(open_grip)
    
    # 2. Go down directly over the cube (fingers still open)
    elif frames < 300:
        target = np.array([0.4, 0.0, 0.025])
        franka.gripper.apply_action(open_grip)
        
    # 3. Pause arm movement and CLOSE fingers firmly
    elif frames < 400:
        target = np.array([0.4, 0.0, 0.025])
        franka.gripper.apply_action(close_grip)
        
    # 4. Lift up (CONTINUE sending close command so it doesn't drop it)
    elif frames < 600:
        target = np.array([0.4, 0.0, 0.3])
        franka.gripper.apply_action(close_grip)
        
    # 5. Move to drop zone (STILL holding close command)
    elif frames < 800:
        target = np.array([0.4, 0.4, 0.3])
        franka.gripper.apply_action(close_grip)
        
    # 6. Lower to drop zone
    elif frames < 950:
        target = np.array([0.4, 0.4, 0.04])
        franka.gripper.apply_action(close_grip)
        
    # 7. OPEN fingers to release
    elif frames < 1050:
        target = np.array([0.4, 0.4, 0.04])
        franka.gripper.apply_action(open_grip)
        
    # 8. Retract up out of the way
    else:
        target = np.array([0.4, 0.4, 0.3])
        franka.gripper.apply_action(open_grip)

    actions = controller.forward(
        target_end_effector_position=target,
        target_end_effector_orientation=orientation
    )
    franka.apply_action(actions)

app.close()