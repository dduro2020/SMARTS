
# import pybullet as p
from smarts.core.utils import pybullet as p
import pybullet_data
import os, inspect

import time
from smarts.core.lidar_sensor_params import BasicLidar
from smarts.core.lidar import Lidar

from smarts.core.chassis import BoxChassis, AckermannChassis
from smarts.core.coordinates import Dimensions, Heading, Pose
from smarts.core.vehicle import VEHICLE_CONFIGS, Vehicle, VehicleState

p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -9.8)
origin = (0,0,1)
lidar = Lidar(origin, BasicLidar)

p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, 0)
chassis = BoxChassis(
    pose=Pose.from_center((3, 1, 0.1), Heading(0)),
    speed=0,
    dimensions=Dimensions(length=3, width=1, height=1),
    bullet_client=p,
)
# chassis = AckermannChassis(
#     pose=Pose.from_center((3, 1, 0.1), Heading(0)),
#     bullet_client=p,
# )

car = Vehicle(
    id="sv-132",
    chassis=chassis,
    vehicle_config_type="passenger",
    visual_model_filepath="smarts/assets/vehicles/visual_model/simple_car.blend",
)
n = 0
try:
    while True:
        p.stepSimulation()
        point_cloud, hits, rays = lidar.compute_point_cloud(bullet_client=p)

        if n == 0:
            time.sleep(1)
            for i, (start, end) in enumerate(rays):
                n = 1
                color = [1, 0, 0] if hits[i] else [0, 1, 0]  # Rojo si impacta, verde si no
                p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=1.5)

        time.sleep(1 / 240)
except KeyboardInterrupt:
    print("\nSimulación detenida por el usuario.")
    p.disconnect()
    
