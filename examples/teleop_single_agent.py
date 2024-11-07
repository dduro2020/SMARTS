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
            v = 15
        elif action == '1':  # Slow down
            v = -15
        elif action == '2':  # Turn left
            v = 1
            w = -0.5
        elif action == '3':  # Turn right
            v = 1
            w = 0.5
        else:
            print("Invalid option. Defaulting to no action.")

        return v, w
    
    def closest_obstacle_warning(self, measurements, aux_measures):
        """
        Encuentra el ángulo de la coordenada (x, y, z) más cercana al origen (0, 0, 0),
        considerando solo las coordenadas en el plano xy para el cálculo del ángulo.
        
        Parameters:
            measurements (np.ndarray): Arreglo de 300 coordenadas 3D (x, y, z).
            aux_measures (np.ndarray): Arreglo con las coordenadas auxiliares para restar del arreglo `measurements`.
        
        Returns:
            None
        """
        # Verificar que el arreglo tenga 300 puntos 3D
        if measurements.shape != (300, 3):
            print("Error: El arreglo debe contener 300 coordenadas 3D.")
            return

        # Eliminar las coordenadas [0, 0, 0] y restar el primer vector de aux_measures
        measurements = measurements[~np.all(measurements == [0, 0, 0], axis=1)] - aux_measures[0]

        # Calcular las distancias al origen (0, 0, 0) para cada punto
        distances = np.linalg.norm(measurements, axis=1)
        
        # Encontrar el índice de la distancia mínima
        min_index = np.argmin(distances)
        
        # Obtener las coordenadas del punto más cercano
        closest_point = measurements[min_index]
        
        # Calcular el ángulo en grados en el plano xy usando atan2
        angle = np.degrees(np.arctan2(closest_point[1], closest_point[0]))  # atan2(y, x)
        
        # Imprimir el resultado
        print(f"El obstáculo más cercano está a {angle:.2f} grados, con una distancia de {distances[min_index]:.2f} unidades.")


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
                print(f"RLidar: {observation['lidar_point_cloud']}")
            episode.record_step(observation, reward, terminated, truncated, info)

            agent.closest_obstacle_warning(observation['lidar_point_cloud']['point_cloud'], observation['lidar_point_cloud']['ray_origin'])

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
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
