from typing import Any, Tuple
import numpy as np
import gymnasium as gym

TARGET_HEADING = -np.pi/2
MAX_ALIGN_STEPS = 19
MAX_STEPS = 500
MAX_DIST = 8.5

class CParkingAgent(gym.Wrapper):
    """Un agente adaptado para estacionamiento utilizando medidas específicas del LIDAR."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Entorno de SMARTS para un solo agente.
        """
        super(CParkingAgent, self).__init__(env)
        self.step_number = 0
        self.last_orientation = 0

        agent_ids = list(env.agent_interfaces.keys())
        assert (
            len(agent_ids) == 1
        ), f"Expected env to have a single agent, but got {len(agent_ids)} agents."
        self._agent_id = agent_ids[0]

        if self.observation_space:
            self.observation_space = self.observation_space[self._agent_id]
        if self.action_space:
            self.action_space = self.action_space[self._agent_id]

    def step(self, action: Any, target: np.ndarray) -> Tuple[Any, float, bool, bool, Any]:
        """Performs a step in the SMARTS environment and calculates the parking reward.

        Args:
            action (Any): Action taken by the agent.
            target (np.ndarray): Target position as a 3-element array (x, y, theta).

        Returns:
            Tuple[Any, float, bool, bool, Any]: Observation, reward, episode end flags, and additional info.
        """
        obs, _, terminated, truncated, info = self.env.step({self._agent_id: action})
        
        agent_obs = obs[self._agent_id]
        
        filtrated_lidar = self.filtrate_lidar(agent_obs["lidar_point_cloud"]["point_cloud"], np.array(agent_obs["ego_vehicle_state"]["position"]), agent_obs["ego_vehicle_state"]["heading"])
        car_pose = np.array(agent_obs["ego_vehicle_state"]["position"])
        heading = agent_obs["ego_vehicle_state"]["heading"]
        car_speed = agent_obs['ego_vehicle_state']['speed']

        reward = self._compute_parking_reward(car_pose, heading, car_speed, target , TARGET_HEADING, filtrated_lidar)

        return (
            agent_obs,
            reward,
            terminated[self._agent_id],
            truncated[self._agent_id],
            info[self._agent_id],
        )

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Any]:
        """Reinicia el entorno de SMARTS y devuelve la observación inicial.

        Returns:
            Tuple[Any, Any]: Observación y datos adicionales.
        """
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[self._agent_id], info[self._agent_id]

    # def _compute_parking_reward(
    #     self,
    #     car_pose: np.ndarray,
    #     car_orient: float,
    #     speed: float,
    #     target_pose: np.ndarray,
    #     target_orient: float,
    #     lidar_data: np.ndarray,
    # ) -> float:
    #     # 1. Distancia al objetivo (relativas)
    #     dist_to_target = np.linalg.norm(target_pose)
    #     horizontal_dist = abs(target_pose[1])
    #     vertical_dist = abs(target_pose[0])

    #     # 2. Diferencia de orientación (normalizada a [-π, π])
    #     orient_diff = np.abs(np.arctan2(
    #         np.sin(car_orient - target_orient),
    #         np.cos(car_orient - target_orient)
    #     ))

    #     # 3. Detección de colisión
    #     min_lidar_dist = np.min(np.linalg.norm(lidar_data, axis=1)) if len(lidar_data) > 0 else np.inf
    #     collision = min_lidar_dist < 0.1

    #     ### ---- Cálculo de recompensas ---- ###
    #     reward = 0.0

    #     # A) Recompensa por distancia (suave y progresiva)
    #     distance_reward = -dist_to_target  # Linear es mejor que sigmoide en este caso
    #     reward += distance_reward * 0.5  # Peso moderado

    #     # B) Recompensa por ORIENTACIÓN (crítica cerca del objetivo)
    #     orientation_weight = 1.0 / (1.0 + 10.0 * dist_to_target)  # Peso aumenta al acercarse
    #     orientation_reward = -orient_diff * orientation_weight
    #     reward += orientation_reward * 2.0  # Peso alto

    #     # C) Recompensa por ALINEACIÓN HORIZONTAL (para aparcamiento lateral)
    #     if horizontal_dist < 0.5:  # Zona donde la alineación horizontal importa
    #         horizontal_alignment_reward = -horizontal_dist * 1.5
    #         reward += horizontal_alignment_reward

    #     # D) Penalización por velocidad (incentivar frenada suave al final)
    #     if dist_to_target < 0.3:
    #         speed_penalty = 0.75 - abs(speed)
    #         print(f"ORIENT_DIFF: {orient_diff} HOR DIST: {target_pose[1]} REWARD: {reward}")
    #         reward += speed_penalty

    #     # E) Gran recompensa por éxito completo
    #     if (dist_to_target < 0.1 and 
    #         orient_diff < 0.1 and 
    #         abs(speed) < 0.1):
    #         print(f"CONSEGUIDO!!, ORIENT: {orient_diff}, HDIST: {horizontal_dist}")
    #         reward = 25.0  # Recompensa final grande

    #     # F) Penalización por colisión
    #     if collision or dist_to_target >= MAX_DIST or target_pose[1] < -0.75:
    #         print(f"COLISION: {collision}, DISTANCIA: {dist_to_target}, DIST H: {target_pose[1]}")
    #         reward = -5.0

    #     return reward
        
    def _compute_parking_reward(
        self,
        car_pose: np.ndarray,
        car_orient: float,
        speed: float,
        target_pose: np.ndarray,
        target_orient: float,
        lidar_data: np.ndarray,
    ) -> float:
        """Calcula la recompensa del aparcamiento basada en la posición, orientación, velocidad y LiDAR."""

        # 1. Distancia al objetivo (el target ya está en relativas)
        dist_to_target = np.linalg.norm(target_pose)
        horizontal_dist = abs(target_pose[1])
        vertical_dist = abs(target_pose[0])
        orient_diff = np.abs(np.arctan2(np.sin(car_orient - target_orient), np.cos(car_orient - target_orient)))

        # Escalar la distancia al objetivo a un rango manejable
        if dist_to_target <= 0.25:
            dist_to_target = 0.2
        if vertical_dist <= 0.2 and horizontal_dist <= 0.1:
            dist_to_target = 0.1
        
        if horizontal_dist <= 0.1:
            horizontal_dist = 0.1
        
        #Recompensa aproximacion
        distance_reward = (1 / (1 + np.exp(3 * (horizontal_dist - 0.25))))
        if orient_diff < 0.25 and horizontal_dist < 0.3:
            distance_reward += (1 / (1 + np.exp(3 * (vertical_dist - 0.25))))/2
            print(f"FASE FINAL: DIST REWARD: {distance_reward}")

        if dist_to_target >= MAX_DIST or target_pose[1] < -0.75: #emula obstaculo horizontal
            distance_reward = -5  # Penalización máxima por distancia
            print(f"Terminado por distancia, HOR DIST: {target_pose[1]}")

        # 2. Recompensa por orientación (escalada a [0, 1])
        
        
        if horizontal_dist < 0.3:
            a = 3.5  # Ajuste de escala
            b = 2.5  # Controla la velocidad de caída exponencial
            c = -0.75  # Límite inferior de penalización

            orientation_reward = a * np.exp(-b * orient_diff) + c
            orientation_reward = max(-0.5, min(orientation_reward, 3))
            # orientation_reward = ((5 * np.pi / 12) / orient_diff) - 3#(0.1/horizontal_dist)
            # orientation_reward = max(-1, min(orientation_reward, 3))

            # orientation_reward = -(((5 * np.pi) / 12) * orient_diff) + 0.4#(0.1/horizontal_dist)
            # orientation_reward = max(-0.5, min(orientation_reward, 1))  # Asegurar rango [-0.5, 1]
            print(f"ORIENT_DIFF: {orient_diff} HOR DIST: {target_pose[1]} REWARD: {orientation_reward}")
            #ñapa para evitar estancamiento
            # if self.last_orientation - orient_diff > 0.05:
            #     orientation_reward += (1-orient_diff)/2
            #     print(f"MEJORA, REWARD: {orientation_reward}")
            # elif orient_diff > 0.2 and abs(self.last_orientation - orient_diff) < 0.01:
            #     orientation_reward -= 0.2
            #     print(f"ESTANCADO, REWARD: {orientation_reward}")
            # else:
            #     orientation_reward += 0.1
            #     print(f"ENDEREZANDO, REWARD: {orientation_reward}")
            if orient_diff > 0.2 and abs(self.last_orientation - orient_diff) < 0.01:
                orientation_reward -= 0.2
                print(f"ESTANCADO, REWARD: {orientation_reward}, LAST: {self.last_orientation} NEW: {orient_diff}")

        else:
            orientation_reward = 0

        if orient_diff > ((5 * np.pi) / 12):  # 75º
            orientation_reward = -5 # Penalización máxima por orientación
            print("Terminado por orientacion")

        # 3. Penalización por velocidad (escalada a [-1, 0])
        if abs(speed) > 2:
            speed_penalty = -0.5  # Penalización máxima por velocidad
        else:
            speed_penalty = 0

        # 4. Bonificación por detenerse correctamente (escalada a [0, 1])
        if orient_diff < 0.1 and horizontal_dist < 0.25 and vertical_dist < 0.25 and abs(speed) < 0.15:
            stopping_bonus = (MAX_STEPS - MAX_ALIGN_STEPS - self.step_number)*2.5# Bonificación máxima por detenerse
            print(f"CONSEGUIDO!!, ORIENT: {orient_diff}, HDIST: {horizontal_dist}")
        else:
            stopping_bonus = 0

        # 5. Penalización por colisión (escalada a [-1, 0])
        min_lidar_dist = np.min(np.linalg.norm(lidar_data, axis=1)) if len(lidar_data) > 0 else np.inf
        if min_lidar_dist < 0.1:
            collision_penalty = -5  # Penalización máxima por colisión
        else:
            collision_penalty = 0

        # 6. Cálculo final de la recompensa (escalada a [-1, 1])
        reward = (
            distance_reward
            + orientation_reward
            + speed_penalty
            + stopping_bonus
            + collision_penalty
        )
        self.last_orientation = orient_diff

        return reward

    
    def filtrate_lidar(self, lidar_data: np.ndarray, car_pose: np.ndarray, heading: float) -> np.ndarray:
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




