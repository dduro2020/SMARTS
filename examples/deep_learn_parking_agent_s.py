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
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

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

    # Calcular puntos relativos
    relative_points = lidar_data_copy - car_pose

    heading_corr = heading + np.pi / 2 # Corrige el sistema respecto el error de SMARTS
        
    # Los puntos absolutos no están en funcion de 0º sino de la orientacion de la carretera

    rotation_matrix = np.array([
        [np.cos(heading_corr), -np.sin(heading_corr), 0],
        [np.sin(heading_corr),  np.cos(heading_corr), 0],
        [0,                0,                1]  # Z no cambia
    ])

    # 3. Aplicar la transformación de rotación
    rotated_points = relative_points @ rotation_matrix.T

    # Convertir heading a grados
    heading_deg = np.degrees(heading)

    num_points = len(lidar_data_copy)
    lidar_resolution = 360 / num_points

    shift = int(round((heading_deg-90) / lidar_resolution))
    # Aplicar el desplazamiento circular
    rotated_lidar = np.roll(rotated_points, shift=shift, axis=0)

    return rotated_lidar

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.init_weights()  # Asegurar inicialización correcta
    
    def init_weights(self):
        """Inicializa los pesos de manera segura para evitar NaN."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.9, alpha=0.001, epsilon=1.0, min_epsilon=0.1, decay_rate=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma  
        self.epsilon = epsilon  
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.batch_size = 32
        self.memory = deque(maxlen=10000)

        self.init_pose = np.array([0,0,0])
        self.parking_target_pose = np.array([0,0,0])
        self.actions = [(0, -0.5), (-1, 0), (0.0, 0.0), (1, 0), (0, 0.5)] 
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Ejecutando en: {self.device}")  # Ver en qué dispositivo corre
        if torch.cuda.is_available():
            print(f"GPU detectada: {torch.cuda.get_device_name(0)}")

        self.model = DQN(state_size, action_size).to(self.device)
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()

        self.check_initial_weights()

    def check_initial_weights(self):
        """Verifica si los pesos de la red contienen NaN tras la inicialización."""
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f" ERROR: El peso {name} contiene NaN tras la inicialización ")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # def act(self, state):
    #     if np.isnan(state).any():
    #         print(" ERROR: `state` contiene NaN antes de pasarlo a la red ")
    #         return random.choice(self.actions)

    #     state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     with torch.no_grad():
    #         q_values = self.model(state_tensor)

    #     if torch.isnan(q_values).any():
    #         print(" ERROR: `q_values` contiene NaN ")
    #         return random.choice(self.actions)

    #     return self.actions[torch.argmax(q_values).item()]
    def act(self, state):
        """Selecciona una acción usando la política Softmax."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state_tensor).squeeze(0)  # Obtener valores Q

        # Aplicar Softmax
        probabilities = F.softmax(q_values, dim=0).cpu().numpy()

        # Elegir acción según distribución Softmax
        action_index = np.random.choice(len(self.actions), p=probabilities)
        return self.actions[action_index]

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor([self.actions.index(a) for a in actions]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (self.gamma * next_q_values * (1 - dones))

        if torch.isnan(q_values).any():
            print(" ERROR: `q_values` en entrenamiento contiene NaN ")
            return

        if torch.isnan(targets).any():
            print(" ERROR: `targets` en entrenamiento contiene NaN ")
            return

        loss = self.loss_fn(q_values, targets.detach())

        if torch.isnan(loss):
            print(" ERROR: `loss` es NaN. Se detiene el entrenamiento ")
            return

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

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
        delta_position = np.array([delta_position[1], delta_position[0], delta_position[2]])
    
        # Los puntos absolutos no están en funcion de 0º sino de la orientacion de la carretera

        rotation_matrix = np.array([
            [np.cos(heading), np.sin(heading), 0],
            [-np.sin(heading),  np.cos(heading), 0],
            [0,                0,                1]  # Z no cambia
        ])
        # # Aplicar la transformación
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
        
        # Distancia euclidiana al objetivo
        distance_to_target = np.linalg.norm(self.parking_target_pose)
        distance_to_target = np.clip(distance_to_target, 0, 1e6)  # Evitar valores extremos
        
        signed_distance_to_target = distance_to_target if self.parking_target_pose[0] > 0 else -distance_to_target
        
        # Diferencia de orientación ajustada a [-pi, pi]
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
        distances = np.nan_to_num(distances, nan=0.0, posinf=1e6, neginf=-1e6)  # Reemplazar NaN e Inf
        
        lidar_resolution = 360 / len(distances)
        index_90 = int(round(90 / lidar_resolution))
        index_270 = int(round(270 / lidar_resolution))
        
        if np.isfinite(distances[index_90]) and np.isfinite(distances[index_270]):
            distance_90 = self.discretize(distances[index_90])
            distance_270 = self.discretize(distances[index_270])
            distance_difference = self.discretize(distance_270 - distance_90)
        else:
            distance_difference = 0  # En vez de Inf, un valor neutro
        
        # Discretizar velocidad para mejor aprendizaje
        discretized_speed = self.discretize(car_speed, step=0.1, max_value=20)

        # Encontrar la distancia mínima válida en el array 'distances'
        valid_distances = distances[distances > 0]
        min_distance = np.min(valid_distances) if valid_distances.size > 0 else 1e6  # Evitar Inf

        # Determinar el ángulo de la distancia mínima
        min_index = np.argmin(distances)
        angle = min_index * lidar_resolution

        # Asignar signo a la distancia mínima según el ángulo
        min_distance_signed = min_distance if 0 <= angle <= 180 else -min_distance
        discretized_min_distance = self.discretize(min_distance_signed, 0.2)

        # Devolver estado asegurando que todos los valores sean finitos
        return (
            self.discretize(signed_distance_to_target),
            self.discretize(heading_error, step=0.1, max_value=np.pi),
            discretized_speed,
            distance_difference,
            discretized_min_distance
        )

    
    def save_q_table(agent, filename="/home/duro/SMARTS/examples/neural-q.log"):
        """Guarda la tabla Q estimada a partir de la memoria del agente."""
        with open(filename, "w") as file:
            file.write("Estado\tValores Q\n")

            for experience in agent.memory:
                state, action, reward, next_state, done = experience

                # Convertir el estado en tensor
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)

                # Obtener valores Q de la red
                with torch.no_grad():
                    q_values = agent.model(state_tensor).cpu().numpy().flatten()

                # Guardar en archivo
                file.write(f"{state} \t {q_values.tolist()}\n")

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
            print("No se detectaron dos vehículos en el point cloud.")
            return None
            
        
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
    agent = DQNAgent(5, 5)
    n_ep = 0

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
        if parking_target == None:
            terminated = True
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

                # Almacenar la experiencia en la memoria
                agent.remember(state, action, reward, next_state, terminated)

                agent.train()

                observation = next_observation
                episode.record_step(observation, reward, terminated, truncated, info)
                if abs(state[0]) > 7.5 or abs(state[1]) > np.pi/3:
                    terminated = True
                    break
        # print(f"Recompensa de episodio: {t_reward}")
        agent.decay_epsilon()
        n_ep = n_ep + 1
        if (n_ep % 100 == 0):
            agent.save_q_table()

    agent.save_q_table()

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
        num_episodes=1000,
        max_episode_steps=200,
    )
