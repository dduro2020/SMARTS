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

AGENT_ID: Final[str] = "Agent"

class KeepLaneAgent(Agent):
    def __init__(self):
        self.state = DirectController()
    

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
            v = 1
        elif action == '1':  # Slow down
            v = -1
        elif action == '2':  # Turn left
            v = 1
            w = -0.5
        elif action == '3':  # Turn right
            v = 1
            w = 0.5
        else:
            print("Invalid option. Defaulting to no action.")

        return v, w

    def compute_parking_reward(self, lidar_data: np.ndarray, car_pose: np.ndarray) -> float:
        """Calcula la recompensa basada en las medidas de LIDAR a 90° y 270°.

        Args:
            lidar_data (np.ndarray): Datos del punto LIDAR alrededor del vehículo.
            car_pose (np.ndarray): Posición actual del agente (coordenadas absolutas).

        Returns:
            float: Recompensa calculada.
        """
        lidar_length = len(lidar_data)
        index_90 = lidar_length // 4  # Índice correspondiente a 90°.
        index_270 = (3 * lidar_length) // 4  # Índice correspondiente a 270°.

        # Asignar 'inf' a los puntos donde no hay obstáculos ([0, 0, 0]).
        lidar_data[np.all(lidar_data == [0, 0, 0], axis=1)] = float('inf')

        relative_lidar = lidar_data - car_pose  # Convertir a coordenadas relativas.
        distances = np.linalg.norm(relative_lidar, axis=1)  # Calcular distancias.

        # Obtener las distancias en los ángulos deseados.
        distance_90 = distances[index_90]
        distance_270 = distances[index_270]
        print(f"Distancia delante: {distance_90}")
        print(f"Distancia detrás: {distance_270}")

        # Si alguno de los valores es infinito, penalización máxima.
        if np.isinf(distance_90) or np.isinf(distance_270):
            return -10.0  # Penalización por falta de datos relevantes.

        # Calcula la diferencia absoluta entre las dos distancias.
        distance_difference = abs(distance_90 - distance_270)

        # Rangos de recompensa según la diferencia de distancias.
        if distance_difference < 0.5:
            reward = 0.0  # Perfecto equilibrio.
        elif distance_difference < 1.0:
            reward = -1.0  # Ligera penalización.
        elif distance_difference < 1.5:
            reward = -3.0  # Penalización moderada.
        else:
            reward = -5.0  # Penalización severa por desbalance.

        return reward
    
    def closest_obstacle_warning(self, measurements, aux_measures):

        if np.any(measurements != 0):
            measurements = measurements[~np.all(measurements == [0, 0, 0], axis=1)] - aux_measures[0]
            distances = np.linalg.norm(measurements, axis=1)
            
            min_index = np.argmin(distances)
            closest_point = measurements[min_index]
            angle = np.degrees(np.arctan2(closest_point[1], closest_point[0]))  # atan2(y, x)
            
            print(f"El obstáculo más cercano está a {angle:.2f} grados, con una distancia de {distances[min_index]:.2f} unidades.")
        else:
            print(f"No hay obstaculos")
    
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

    def get_state(self, observation):
        """Extrae y discretiza el estado basado en el LiDAR y la posición del vehículo."""
        lidar_data = observation["lidar_point_cloud"]["point_cloud"]
        lidar_data_copy = np.copy(lidar_data)
        car_pose = np.array(observation["ego_vehicle_state"]["position"])
        heading = observation["ego_vehicle_state"]["heading"]

        # Procesar datos del LiDAR
        lidar_data_copy[np.all(lidar_data_copy == [0, 0, 0], axis=1)] = float('inf')  # Asignar inf a obstáculos ausentes.
        relative_points = lidar_data_copy - car_pose
        distances = np.linalg.norm(relative_points, axis=1)

        # Obtener distancias discretizadas a 90° y 270°
        lidar_length = len(distances)
        index_90 = lidar_length // 4
        index_270 = (3 * lidar_length) // 4

        # heading_deg = np.degrees(heading) % 360
        # index_90 = int((heading_deg) % 360)
        # index_270 = int((heading_deg + 180) % 360)

        distance_90 = self.discretize(distances[index_90])
        distance_270 = self.discretize(distances[index_270])

        # Calcular la diferencia discretizada
        distance_difference = self.discretize(abs(distance_90 - distance_270))

        # Retornar el estado como un valor inmutable (float o tupla)
        print(f"Diferencia de distancias: {distance_difference}")
        return float(distance_difference)


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

        terminated = False
        while not terminated:
            action = agent.act(observation)

            observation, reward, terminated, truncated, info = env.step(action)
            # Extracting linear vel and position
            print(f"Actual Speed: {observation['ego_vehicle_state']['speed']}")
            print(f"Actual Pos: {observation['ego_vehicle_state']['position']}")
            print(f"Actual Heading: {observation['ego_vehicle_state']['heading']}")
            # print(observation)
            resp = input("Printig point_cloud? (yes/no): ")
            if resp == "yes":
                print(f"RLidar: {observation['lidar_point_cloud']['point_cloud']}")
            episode.record_step(observation, reward, terminated, truncated, info)
            # agent.compute_parking_reward(observation['lidar_point_cloud']['point_cloud'], observation['ego_vehicle_state']['position'])
            # agent.closest_obstacle_warning(observation['lidar_point_cloud']['point_cloud'], observation['lidar_point_cloud']['ray_origin'])
            agent.get_state(observation)

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
