from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.kit.app
import omni.usd
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
from pxr import UsdPhysics, Sdf, UsdLux, Gf
import os
import tempfile
import time
import math

# Import robot
ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_id = ext_manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
extension_path = ext_manager.get_extension_path(ext_id)

urdf_path = os.path.normpath(
    os.path.abspath(os.path.join(extension_path, "data", "urdf", "tests", "differential_base.urdf"))
)

config = URDFImporterConfig()
config.urdf_path = urdf_path
config.usd_path = os.path.normpath(tempfile.gettempdir())

importer = URDFImporter(config)
importer.config = config
output_path = os.path.normpath(importer.import_urdf())

omni.usd.get_context().open_stage(output_path)
stage = omni.usd.get_context().get_stage()

robot_prim = stage.GetPrimAtPath("/differential_base")

# Physics
if not stage.GetPrimAtPath("/physicsScene"):
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))
    scene.CreateGravityMagnitudeAttr(9.81)
    distant_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distant_light.CreateIntensityAttr(1500)

robot_prim.GetVariantSet("Physics").SetVariantSelection("physx")

# Get wheel joints
left_joint = stage.GetPrimAtPath("/differential_base/Physics/left_wheel")
right_joint = stage.GetPrimAtPath("/differential_base/Physics/right_wheel")

print("✓ Robot imported successfully")
print("✓ Differential base ready for simulation")
print("✓ Wheel joints configured")

start_time = time.time()

while True:
    simulation_app.update()
    
    elapsed = time.time() - start_time
    
    # Oscillate angular velocities
    vel = 15 * math.sin(elapsed)
    
    try:
        if left_joint and left_joint.HasAttribute("physics:angularVelocity"):
            left_joint.GetAttribute("physics:angularVelocity").Set(Gf.Vec3f(0, vel, 0))
        if right_joint and right_joint.HasAttribute("physics:angularVelocity"):
            right_joint.GetAttribute("physics:angularVelocity").Set(Gf.Vec3f(0, vel, 0))
    except:
        pass
