import argparse
import logging
import random
import sys
import warnings
import numpy as np
from pathlib import Path
from typing import Final, Any

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")

import gymnasium as gym
from smarts.env.gymnasium.wrappers.complex_parking_agent import CParkingAgent

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, ActionSpaceType, AgentType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.core.scenario import Scenario

import time
import random

from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

MAX_ALIGN_STEPS = 19

AGENT_ID: Final[str] = "Agent"

TARGET_HEADING = -np.pi/2
# ROAD: -pi/2
# DEST: [28.16 96.8   0.  ]

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


class LearningAgent:
    """Agente de Q-learning optimizado para aparcamiento, que utiliza LiDAR con velocidad angular fija."""

    def __init__(self, epsilon=0.99, min_epsilon=0, decay_rate=0.99, alpha=0.2, gamma=0.9):
        self.epsilon = epsilon  # Probabilidad inicial de exploración
        self.min_epsilon = min_epsilon  # Valor mínimo de epsilon
        self.decay_rate = decay_rate  # Tasa de decremento para epsilon
        self.alpha = alpha  # Tasa de aprendizaje
        self.gamma = gamma  # Factor de descuento
        self.init_pose = np.array([0,0,0])
        self.parking_target_pose = np.array([0,0,0])
        self.q_table = {} # Tabla Q para almacenar los valores de estado-acción
        self.actions = [(0, -0.5),(-1, 0),(0.0, 0.0),(1, 0),(0, 0.5)]  # Posibles acciones

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

        # Matriz de rotación en el plano XY (Z no cambia)
        rotation_matrix = np.array([
            [np.cos(-heading), -np.sin(-heading), 0],
            [np.sin(-heading),  np.cos(-heading), 0],
            [0,                0,                1]  # La Z se mantiene igual
        ])

        # Aplicar la transformación
        pose_relative = rotation_matrix @ delta_position

        return pose_relative

    def get_state(self, observation, target_pose):
        """Extrae y discretiza el estado basado en la posición, orientación, velocidad y LiDAR del vehículo."""
        
        # Extraer información relevante
        car_position = np.array(observation["ego_vehicle_state"]["position"])
        car_heading = observation["ego_vehicle_state"]["heading"]
        car_speed = observation["ego_vehicle_state"]["speed"]
        
        # Calcular la posición relativa del vehículo con respecto a init_pose
        car_pose_relative = self.get_relative_coordinates(car_position, car_heading)
        
        # Sumar la posición relativa del vehículo con target_pose (relativo a init_pose)
        self.parking_target_pose = target_pose - car_pose_relative
        
        # Distancia euclidiana al objetivo (target_pose ahora es relativa al vehículo)
        distance_to_target = np.linalg.norm(self.parking_target_pose)
        if self.parking_target_pose[0] > 0:
            signed_distance_to_target = distance_to_target  # Positiva si está delante
        else:
            signed_distance_to_target = -distance_to_target
        
        # Diferencia de orientación (ajustada a [-pi, pi])
        heading_error = np.arctan2(
            np.sin(TARGET_HEADING - car_heading), 
            np.cos(TARGET_HEADING - car_heading)
        )
        
        # Filtrar datos del LiDAR
        filtered_lidar = filtrate_lidar(
            observation["lidar_point_cloud"]["point_cloud"], 
            car_position, 
            car_heading
        )
        
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
            distance_difference,
            discretized_min_distance
        )



    def choose_action(self, state):
        """Selecciona una acción basada en la política epsilon-greedy."""
        if np.random.rand() < self.epsilon:
            # Explorar: Elegir una acción aleatoria
            return random.choice(self.actions)

        # Explotar: Elegir la mejor acción conocida
        # print(f"Diferencia de distancias: {state}")
        if state not in self.q_table:
            # Inicializar valores Q para acciones en el estado si no existen
            self.q_table[state] = {action: 0.0 for action in self.actions}

        # Devolver la acción con el valor Q más alto en este estado
        return max(self.q_table[state], key=self.q_table[state].get)

    def act(self, state):
        """Genera una acción basada en la observación del entorno."""
        # state = self.get_state(observation, target_pose)
        action = self.choose_action(state)
        # print(f"Accion elegida: {action}       En estado: {state}")
        return np.array(action)

    def learn(self, state, action, reward, next_state):
        """Actualiza la tabla Q según la fórmula de Q-learning."""
        state = tuple(state)
        next_state = tuple(next_state)

        action = tuple(action)

        # Inicializar los estados en la tabla Q si no existen
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in self.actions}

        # Obtener el valor Q futuro máximo
        max_future_q = max(self.q_table[next_state].values(), default=0.0)

        # Actualizar la tabla Q usando la ecuación de Q-learning
        current_q = self.q_table[state][action]
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - current_q)

    def decay_epsilon(self):
        """Reduce epsilon según la tasa de decremento."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)
    
    def print_q_table(self):
        """Imprime la tabla Q completa."""
        for state, actions in self.q_table.items():
            print(f"State: {state}")
            for action, value in actions.items():
                print(f"  Action: {action}, Q-value: {value}")

    def move_to_random_position(self, current_position, target_position, accelerate, steps, first_act):
        """Mueve el vehículo a una posición (target)."""

        distance = target_position - current_position
        action = 0

        # Determinar si avanzar o retroceder
        if accelerate == True:
            action = 10 if distance > 0 else -10

        # Paramos si estamos cerca o si llegamos a las maximas steps
        if abs(distance) < 0.25 or steps == MAX_ALIGN_STEPS:
            # print(f"finished, current pose: {current_position}")
            action = -first_act
                
        return np.array([action, 0.0])

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
        filtrated_lidar = filtrate_lidar(observation["lidar_point_cloud"]["point_cloud"], pose, observation["ego_vehicle_state"]["heading"])

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
        
        return midpoint



def main(scenarios, headless, num_episodes=200, max_episode_steps=None):
    agent_interface = AgentInterface(
        action=ActionSpaceType.Direct,
        # max_episode_steps=max_episode_steps,
        max_episode_steps=200,
        neighborhood_vehicle_states=True,
        # waypoint_paths=True,
        # road_waypoints=True,
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

    env = CParkingAgent(env)
    agent = LearningAgent()

    for episode in episodes(n=num_episodes):
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        # np.random.seed(int(time.time()))
        random_offset = np.random.choice([-1, 0, 1, 2])
        actual_pose = observation["ego_vehicle_state"]["position"][0]
        target =  actual_pose + random_offset

        # print(f"Moving from pose: {actual_pose} to pose: {target}")

        terminated = False
        moved = False
        
        accelerate = True
        first_action = np.array([0.0, 0.0])
        n_steps = 0
        # parking_target = np.array([0.0, 0.0, 0.0])
        parking_target = agent.find_closest_corners(observation)
        # print(f"El target se encuentra en: {parking_target}")
        # t_reward = 0
        while not terminated:                
            # Mover a posicion aleatoria
            if not moved:
                # Indice 0 es el que se usa en nuestro escenario
                action = agent.move_to_random_position(observation["ego_vehicle_state"]["position"][0], target, accelerate, n_steps, first_action[0])
                accelerate = False
                
                if action[0] + first_action[0] == 0:
                    moved = True

                if n_steps == 0:                    
                    first_action = action
                    
                observation, _, terminated, _, _ = env.step((action[0],action[1]), parking_target)
                n_steps = n_steps + 1
                # print(observation['ego_vehicle_state']['heading'])
                # print(observation["ego_vehicle_state"]["position"])
            
            # Tenemos que asegurarnos que SIEMPRE gastamos MAX_ALIGN_STEPS steps, así no modificamos el entrenamiento
            elif n_steps <= MAX_ALIGN_STEPS:
                # print(observation['ego_vehicle_state']['speed'])
                observation, _, terminated, _, _ = env.step((0.0,0.0), parking_target)
                n_steps = n_steps + 1

            # Nos quedan TOTAL_STEPS-MAX_ALIGN_STEPS para el entrenamiento, SIEMPRE las mismas
            else:
                state = agent.get_state(observation, parking_target)
                action = agent.act(state)

                next_observation, reward, terminated, truncated, info = env.step((action[0],action[1]), agent.parking_target_pose)
                next_state = agent.get_state(next_observation, parking_target)

                agent.learn(state, action, reward, next_state)

                observation = next_observation
                # t_reward = t_reward + reward
                episode.record_step(observation, reward, terminated, truncated, info)
        # print(f"Recompensa de episodio: {t_reward}")
        agent.decay_epsilon()
    agent.print_q_table()

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
        num_episodes=1500,
        max_episode_steps=200,
    )
