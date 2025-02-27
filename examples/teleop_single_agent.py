import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Final

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

import gymnasium as gym
import numpy as np

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))

from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, ActionSpaceType, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.sstudio.scenario_construction import build_scenarios

from smarts.core.controllers.direct_controller import DirectController

from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

AGENT_ID: Final[str] = "Agent"
TARGET_HEADING = -np.pi/2

import matplotlib.pyplot as plt

def plot_scenario(lidar_data: np.ndarray, target_pose: np.ndarray, real_target_pose: np.ndarray):
    """
    Muestra la nube de puntos del LiDAR en 3D, una línea hasta el objetivo y el punto objetivo como una "X".

    Parámetros:
        lidar_data (np.ndarray): Un array de shape (N, 3) que contiene las coordenadas (x, y, z) de los puntos LiDAR.
        target_pose (np.ndarray): Un array de shape (3,) que representa la posición del objetivo (x, y, z).
    """
    # Crear la figura y el eje en 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la nube de puntos del LiDAR
    if lidar_data.size > 0:  # Verificar que hay datos LiDAR
        ax.scatter(lidar_data[:, 0], lidar_data[:, 1], lidar_data[:, 2], c='blue', label='LiDAR Points', s=10)

    # Graficar la línea desde el origen (0, 0, 0) hasta el objetivo (target_pose)
    ax.plot([0, target_pose[0]], [0, target_pose[1]], [0, target_pose[2]], 'r--', label='Target Vector', linewidth=2)

    # Graficar el punto objetivo como una "X"
    ax.scatter(target_pose[0], target_pose[1], target_pose[2], c='red', marker='x', s=100, label='Target Pose')
    ax.scatter(real_target_pose[0], real_target_pose[1], real_target_pose[2], c='black', marker='x', s=100, label='Real Target Pose')

    ax.set_xlim([-10, 10])  # Límite del eje x
    ax.set_ylim([-10, 10])  # Límite del eje y
    ax.set_zlim([0, 1.5])
    # Configuraciones adicionales del gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('LiDAR Data and Target Pose in 3D')
    ax.legend()
    ax.grid(True)

    # Mostrar el gráfico
    plt.show()

def filtrate_lidar(lidar_data: np.ndarray, car_pose: np.ndarray, heading: float) -> np.ndarray:
    """
    Transforma los puntos LIDAR para que sean relativos al vehículo, con el índice 0 a 90° a la izquierda del agente.

    Args:
        lidar_data (np.ndarray): Datos del LIDAR en coordenadas absolutas.
        car_pose (np.ndarray): Posición actual del vehículo en coordenadas absolutas.
        heading (float): Ángulo de orientación del vehículo en radianes.

    Returns:
        np.ndarray: Datos LIDAR transformados en coordenadas relativas.
    """
    lidar_data_copy = np.copy(lidar_data)
    
    # Asignar 'inf' a los puntos inválidos (donde todo es [0, 0, 0])
    lidar_data_copy[np.all(lidar_data_copy == [0, 0, 0], axis=1)] = float('inf')

    # Reordenar de (y, x, z) a (x, y, z)
    # lidar_data_copy = lidar_data_copy[:, [1, 0, 2]]
    
    # Calcular puntos relativos en el nuevo formato
    relative_points = car_pose - lidar_data_copy# - car_pose
    relative_points = relative_points[:, [1, 0, 2]]

    # Establecer todos los valores de z finitos (no inf, -inf ni NaN) a 0
    relative_points[np.isfinite(relative_points[:, 2]), 2] = 0
    
    # Matriz de rotación en el sistema dextrógiro
    rotation_matrix = np.array([
        [ np.cos(heading), np.sin(heading), 0],  # x'
        [-np.sin(heading), np.cos(heading), 0],  # y'
        [0,                0,               1]  # z no cambia
    ])

    # Aplicar la transformación de rotación
    rotated_points = relative_points @ rotation_matrix.T

    # Convertir heading a grados
    heading_deg = np.degrees(heading)

    num_points = len(lidar_data_copy)
    lidar_resolution = 360 / num_points

    shift = int(round((heading_deg - 90) / lidar_resolution))
    # Aplicar el desplazamiento circular
    rotated_lidar = np.roll(rotated_points, shift=shift, axis=0)

    return rotated_lidar

class KeepLaneAgent(Agent):
    def __init__(self):
        self.state = DirectController()
        self.init_pose = np.array([0,0,0])
        self.parking_target_pose = np.array([0,0,0])
    

    def get_user_input(self):
        print("Select option:")
        print("0: Accelerate")
        print("1: Go back")
        print("2: Turn left")
        print("3: Turn right")
        
        choice = input("Option number: ")
        return choice

    def act(self, obs, **kwargs):
        v, w = 0.0, 0.0  # Default action

        action = self.get_user_input()

        if action == '0':  # Accelerate
            v = 5
        elif action == '1':  # Slow down
            v = -5
        elif action == '2':  # Turn left
            v = 0
            w = -0.5
        elif action == '3':  # Turn right
            v = 0
            w = 0.5
        else:
            print("Invalid option. Defaulting to no action.")

        return v, w
    
    def discretize(self, value, step=0.25, max_value=10.0):
        """Discretiza un valor continuo al múltiplo más cercano de 'step'.

        Args:
            value (float): Valor continuo a discretizar.
            step (float): Tamaño del intervalo de discretización.
            max_value (float): Límite máximo (los valores mayores se limitan).

        Returns:
            float: Valor discretizado al múltiplo más cercano de 'step'.
        """
        # Limitar el valor a [-max_value, max_value]
        value = min(max(value, -max_value), max_value)
        # Redondear al múltiplo más cercano de step
        return round(value / step) * step

    def get_state(self, observation, target_pose):
        """Extrae y discretiza el estado basado en la posición, orientación, velocidad y LiDAR del vehículo."""
        
        # Extraer información relevante
        car_position = np.array(observation["ego_vehicle_state"]["position"])
        car_heading = observation["ego_vehicle_state"]["heading"]
        car_speed = observation["ego_vehicle_state"]["speed"]
        
        # Calcular la posición relativa del vehículo con respecto a init_pose
        car_pose_relative = self.get_relative_coordinates(car_position, TARGET_HEADING)
        print(f"CAR RELATIVE (0,0,0): {car_pose_relative}, CAR ABS: {car_position}, CAR INIT: {self.init_pose}")
        
        # Sumar la posición relativa del vehículo con target_pose (relativo a init_pose)
        self.parking_target_pose = target_pose - car_pose_relative
        dist_to_target = np.linalg.norm(self.parking_target_pose)
        # print(f"Distancia a hueco: {dist_to_target}")
        print(f"TARGET RELATIVE: {self.parking_target_pose}, TARGET INIT: {target_pose}")
        
        # Distancia euclidiana al objetivo (target_pose ahora es relativa al vehículo)
        distance_to_target = np.linalg.norm(self.parking_target_pose)
        if self.parking_target_pose[1] > 0:
            signed_distance_to_target = distance_to_target  # Positiva si está delante
        else:
            signed_distance_to_target = -distance_to_target
        
        # Diferencia de orientación (ajustada a [-pi, pi])
        heading_error = np.arctan2(
            np.sin(TARGET_HEADING - car_heading), 
            np.cos(TARGET_HEADING - car_heading)
        )
        print(f"Heading error: {heading_error}")
        
        # Filtrar datos del LiDAR
        filtered_lidar = filtrate_lidar(
            observation["lidar_point_cloud"]["point_cloud"], 
            car_position, 
            TARGET_HEADING
        )
        # print("Filtrado: ")
        # print(filtered_lidar)

        reward = self._compute_parking_reward(car_position, car_heading, car_speed, self.parking_target_pose , TARGET_HEADING, filtered_lidar)
        resp = input("Plot point_cloud? (yes/no): ")
        if resp == "yes":
            plot_scenario(filtered_lidar, self.parking_target_pose, target_pose)
        print(f"RECOMPENSA: {reward}")
        
        distances = np.linalg.norm(filtered_lidar, axis=1)
        lidar_resolution = 360 / len(distances)
        index_90 = int(round(90 / lidar_resolution))
        index_270 = int(round(270 / lidar_resolution))
        if np.isfinite(distances[index_90]) and np.isfinite(distances[index_270]):
            distance_90 = self.discretize(distances[index_90])
            distance_270 = self.discretize(distances[index_270])
            distance_difference = self.discretize(distance_270 - distance_90)
        else:
            distance_difference = np.inf
        
        
        # Discretizar velocidad para mejor aprendizaje
        discretized_speed = self.discretize(car_speed, step=0.1, max_value=20)

        # Encontrar la distancia mínima no nula directamente en el array 'distances'
        min_distance = np.min(distances[distances > 0]) if np.any(distances > 0) else np.inf

        # Si se encontró una distancia válida
        if min_distance != np.inf:
            # Obtener el índice de la distancia mínima en el array original
            min_index = np.argmin(distances)

            # Calcular el ángulo correspondiente a ese índice
            angle = min_index * lidar_resolution

            # Asignar signo según el ángulo (delante: positivo, detrás: negativo)
            if 0 <= angle <= 180:
                min_distance_signed = min_distance  # Positivo si está delante
            else:
                min_distance_signed = -min_distance  # Negativo si está detrás

            # Discretizar la distancia mínima con el signo
            discretized_min_distance = self.discretize(min_distance_signed, 0.2)
        else:
            # Si no hay distancias válidas, asignamos infinito
            discretized_min_distance = np.inf

        return (
            self.discretize(distance_to_target), 
            self.discretize(heading_error, step=0.1, max_value=np.pi), 
            discretized_speed, 
            # distance_difference,
            discretized_min_distance
        )

    def _compute_parking_reward(self, car_pose: np.ndarray, car_orient: float, speed: float, 
                            target_pose: np.ndarray, target_orient: float, lidar_data: np.ndarray) -> float:
        """Calcula la recompensa del aparcamiento basada en la posición, orientación, velocidad y LiDAR."""

        # 1. Distancia al objetivo (el target ya está en relativas)
        dist_to_target = np.linalg.norm(target_pose)
        # print("DISTANCIA: ", dist_to_target)
        if dist_to_target < 0.1:
            dist_to_target = 0.1
        
        
        # distance_reward = (1 / (dist_to_target**2)) # Recompensa pronunciada cuanto más cerca
        distance_reward = (1 / dist_to_target)
        if dist_to_target > 6.5:
            distance_reward = -10
            # print("Terminado por distancia")
        # print(f"Distancia a hueco: {dist_to_target}")

        # 2. Recompensa por orientación (solo si está cerca del parking)
        orient_diff = np.abs(np.arctan2(np.sin(car_orient - target_orient), np.cos(car_orient - target_orient)))
        if dist_to_target < 0.25:
            orientation_reward = max(0, 1 - orient_diff / np.pi) * 50  # Máx: 50, Mín: 0
        else: 
            orientation_reward = 0
            
        if orient_diff > np.pi/2:
            orientation_reward = -10
            # print("Terminado por orientacion")
                
        

        # 3. Penalización por velocidad
        if abs(speed) > 2.5:
            speed_penalty = -10
        else:
            speed_penalty = 0

         # 4. Bonificación por detenerse correctamente estando alineado
        if orient_diff < 0.1 and dist_to_target < 0.1 and abs(speed) < 0.1:
            stopping_bonus = 200
            print("CONSEGUIDO!!")
        else:
            stopping_bonus = 0

        # 5. Penalización por colisión (usando la menor distancia del LiDAR)
        min_lidar_dist = np.min(np.linalg.norm(lidar_data, axis=1)) if len(lidar_data) > 0 else np.inf
        if min_lidar_dist < 0.1:
            collision_penalty = -10
        else:
            collision_penalty = 0

        # 6. Cálculo final de la recompensa
        print(f"Recp dist: {distance_reward}, Recp orient: {orientation_reward}, Recp speed: {speed_penalty}, Recp stop: {stopping_bonus}, Recp collision: {collision_penalty}")
        reward = (
            distance_reward 
            + orientation_reward 
            + speed_penalty 
            + stopping_bonus 
            + collision_penalty
        )

        return reward

    def get_relative_coordinates(self, position, heading):
        """
        Transforma una posición global a coordenadas relativas usando la orientación (heading) en 3D.
        
        Parámetros:
            position (array-like): Coordenadas [x, y, z] en el sistema global.
            heading (float): Ángulo de orientación en radianes.

        Retorna:
            np.ndarray: Coordenadas [x', y', z'] en el sistema relativo.
        """
        init_position = np.array(self.init_pose)

        # Calcular el desplazamiento en coordenadas globales
        delta_position = position - init_position
        delta_position = np.array([delta_position[0], -delta_position[1], delta_position[2]])

    
        # Los puntos absolutos no están en funcion de 0º sino de la orientacion de la carretera

        rotation_matrix = np.array([
            [np.cos(heading), np.sin(heading), 0],
            [-np.sin(heading),  np.cos(heading), 0],
            [0,                0,                1]  # Z no cambia
        ])
        # # Aplicar la transformación
        pose_relative = rotation_matrix @ delta_position

        return delta_position

    def find_closest_corners(self, observation, eps=3, min_samples=10):
        """
        Encuentra las esquinas más cercanas de dos vehículos que delimitan un hueco de aparcamiento a partir de un point cloud.
        
        :param point_cloud: Nube de puntos del escenario (Nx3 numpy array)
        :param eps: Parámetro de distancia para el algoritmo DBSCAN (tolerancia en la agrupación de puntos).
        :param min_samples: Número mínimo de puntos para que un conjunto sea considerado un clúster (DBSCAN).
        
        :return: Punto medio entre las esquinas más cercanas (numpy array con las coordenadas [x, y, z]).
        """

        pose = np.array(observation["ego_vehicle_state"]["position"])
        self.init_pose = pose
        filtrated_lidar = filtrate_lidar(observation["lidar_point_cloud"]["point_cloud"], self.init_pose, TARGET_HEADING)

        # 1. Filtrar los puntos que representan ruido o valores infinitos
        filtrated_lidar = filtrated_lidar[~np.isnan(filtrated_lidar).any(axis=1)]  # Elimina NaNs
        filtrated_lidar = filtrated_lidar[~np.isinf(filtrated_lidar).any(axis=1)]  # Elimina infs
        
        # 2. Aplicar DBSCAN para dividir el point cloud en dos grupos (vehículos)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(filtrated_lidar)
        
        
        # Obtener las etiquetas de los clusters y filtramos solo aquellos que tienen más de 10 puntos
        labels = clustering.labels_

        # plot_clusters(point_cloud, labels)
        
        # Asegurarnos de que hay al menos dos clústeres
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            raise ValueError("No se detectaron dos vehículos en el point cloud.")
        
        # Dividir los puntos en dos grupos (vehículos)
        group_1 = filtrated_lidar[labels == unique_labels[0]]
        group_2 = filtrated_lidar[labels == unique_labels[1]]
        
        # 3. Encontrar las esquinas más cercanas entre los dos vehículos
        closest_distance = np.inf
        closest_pair = None
        
        for corner_1 in group_1:
            for corner_2 in group_2:
                distance = np.linalg.norm(corner_1 - corner_2)  # Distancia euclidiana
                if distance < closest_distance:
                    closest_distance = distance
                    closest_pair = (corner_1, corner_2)
        
        # 4. Calcular el punto medio entre las dos esquinas más cercanas
        if closest_pair is None:
            raise ValueError("No se pudo encontrar un par de esquinas cercanas.")
        
        corner_1, corner_2 = closest_pair
        midpoint = (corner_1 + corner_2) / 2
        midpoint[2] = 0
        midpoint[1] = midpoint[1] + 0.5 # Le sumamos 1/2 ancho del coche para centrar aparcamiento
        
        return midpoint


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_interface = AgentInterface(
        action=ActionSpaceType.Direct,
        max_episode_steps=max_episode_steps,
        neighborhood_vehicle_states=True,
        waypoint_paths=True,
        road_waypoints=True,
        drivable_area_grid_map=True,
        occupancy_grid_map=True,
        top_down_rgb=True,
        lidar_point_cloud=True,
        accelerometer=True,
        lane_positions=True,
        signals=True,
    )


    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=headless,
    )
    env = SingleAgent(env)

    for episode in episodes(n=num_episodes):
        agent = KeepLaneAgent()
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)
        parking_target = agent.find_closest_corners(observation)

        terminated = False
        while not terminated:
            state = agent.get_state(observation, parking_target)
            action = agent.act(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            # Extracting linear vel and position
            print(f"Actual Speed: {observation['ego_vehicle_state']['speed']}")
            # print(f"Actual Pos: {observation['ego_vehicle_state']['position']}")
            # print(f"Target Pos: {parking_target}")
            print(f"Actual Heading: {observation['ego_vehicle_state']['heading']}")
            # print(observation)
            # resp = input("Printig point_cloud? (yes/no): ")
            # if resp == "yes":
            #     print(f"RLidar: {observation['lidar_point_cloud']['point_cloud']}")

            episode.record_step(observation, reward, terminated, truncated, info)
            # agent.compute_parking_reward(observation['lidar_point_cloud']['point_cloud'], observation['ego_vehicle_state']['position'])
            # agent.closest_obstacle_warning(observation['lidar_point_cloud']['point_cloud'], observation['lidar_point_cloud']['ray_origin'])

    env.close()

if __name__ == "__main__":
    parser = minimal_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "loop"),
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "figure_eight"),
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=100,
        max_episode_steps=200,
    )
