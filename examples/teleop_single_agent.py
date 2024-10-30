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

    def modificar_posicion_inicial(self, agent_state):
        # Modificar la posición y la velocidad inicial del agente
        agente_pos_inicial = (7, 0, 0)  # Inicializar en (x=0, y=0, z=0)
        agente_vel_inicial = 0.0  # Velocidad inicial de 5 m/s

        agent_state['ego_vehicle_state']['position'] = agente_pos_inicial
        agent_state['ego_vehicle_state']['speed'] = agente_vel_inicial
        print(f"Agente colocado en {agente_pos_inicial} con velocidad {agente_vel_inicial} m/s.")

        return agent_state
    
    def colocar_vehiculo_delante(self, env, agente_pos, distancia=3):
        # Obtener la posición del agente en el eje X
        x, y, z = agente_pos
        # Colocar el nuevo vehículo 3 metros delante en el eje X
        vehiculo_pos = (x + distancia, y, z)
        
        # Crear el vehículo delante
        nuevo_vehiculo = {
            "veh_id": "vehiculo_delante",
            "vehicle_type": "car",
            "pose": vehiculo_pos
        }
        
        # Aquí puedes definir qué hacer con el vehículo, según lo permita el entorno SMARTS.
        # Por ejemplo, si tu entorno tiene la opción de tráfico predefinido, lo añades allí.
        print(f"Vehículo añadido delante del agente en {vehiculo_pos}.")

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
    
    import numpy as np

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

        observation = agent.modificar_posicion_inicial(observation)
        
        # Colocar un vehículo delante del agente a 3 metros
        agent.colocar_vehiculo_delante(env, observation['ego_vehicle_state']['position'], distancia=3)

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
