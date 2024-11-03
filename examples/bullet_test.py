
import pybullet as p
import pybullet_data
import os, inspect

import time
from smarts.core.lidar_sensor_params import BasicLidar
from smarts.core.lidar import Lidar

ray_from = [0, 0, 1]
ray_to = [0, 0, 0]

def use_ray():
    return p.rayTest(ray_from, ray_to)

p.connect(p.GUI)
p.resetSimulation()
p.setGravity(0, 0, -9.8)
origin = (0,0,0.5)
lidar = Lidar(origin, BasicLidar)

p.loadURDF(os.path.join(pybullet_data.getDataPath(), "plane.urdf"), 0, 0, 0)
cube_id = p.loadURDF(os.path.join(pybullet_data.getDataPath(),"r2d2.urdf"), [3, 1, 0.1])

n = 0
try:
    while True:
        point_cloud, hits, rays = lidar.compute_point_cloud(bullet_client=p)

        # Dibujar cada rayo en PyBullet
        if n == 0:
            for i, (start, end) in enumerate(rays):
                n = 1
                color = [1, 0, 0] if hits[i] else [0, 1, 0]  # Rojo si impacta, verde si no
                p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=1.5)

        p.stepSimulation()
        time.sleep(1 / 240)
except KeyboardInterrupt:
    print("\nSimulación detenida por el usuario.")
    p.disconnect()
    
