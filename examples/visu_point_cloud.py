import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial.distance import cdist


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
    # Asignar 'inf' a los puntos inválidos (donde todo es [0, 0, 0])
    lidar_data[np.all(lidar_data == [0, 0, 0], axis=1)] = float('inf')

    # Calcular puntos relativos
    relative_points = lidar_data - car_pose
    relative_points = relative_points[::-1]

    # Convertir heading a grados
    heading_deg = np.degrees(heading)

    num_points = len(lidar_data)
    lidar_resolution = 360 / num_points

    shift = int(round((heading_deg-90) / lidar_resolution))
    # Aplicar el desplazamiento circular
    rotated_lidar = np.roll(relative_points, shift=shift, axis=0)

    return rotated_lidar

def discretize(value, step=0.25, max_value=10.0):
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

def print_distance(lidar_data: np.ndarray, car_pose: np.ndarray, heading: float):
        filtrate_l = filtrate_lidar(lidar_data, car_pose, heading)
        # print(filtrate_l)
        distances = np.linalg.norm(filtrate_l, axis=1)  # Calcular distancias.

        # Obtener las distancias en los ángulos deseados.
        lidar_resolution = 360 / len(distances)
        index_90 = int(round(90 / lidar_resolution))
        index_270 = int(round(270 / lidar_resolution))
        distance_90 = distances[index_90]
        distance_270 = distances[index_270]
        print(f"Distancia delante: {distance_90}")
        print(f"Distancia detras: {distance_270}")

        # x = filtrate_l[:, 0]
        # y = filtrate_l[:, 1]
        # z = filtrate_l[:, 2]
        # # Crear una figura y un eje 3D
        # fig = plt.figure(figsize=(10, 7))
        # ax = fig.add_subplot(111, projection='3d')

        # # Graficar los puntos
        # ax.scatter(x, y, z, c='b', marker='o', s=10)
        # ax.set_xlim([-10, 10])  # Límite del eje x
        # ax.set_ylim([-10, 10])  # Límite del eje y
        # ax.set_zlim([0, 1.5])    # Límite del eje z
        # # Etiquetas y título
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('Nube de Puntos Lidar')

        # # Mostrar el gráfico
        # plt.show()
def get_state(filtered_lidar, target_pose, target_heading, car_heading, car_speed):
    """Extrae y discretiza el estado basado en la posición, orientación, velocidad y LiDAR del vehículo."""
    
    # Distancia euclidiana al objetivo
    distance_to_target = np.linalg.norm(target_pose)
    
    # Diferencia de orientación (ajustada a [-pi, pi])
    heading_error = np.arctan2(
        np.sin(target_heading - car_heading), 
        np.cos(target_heading - car_heading)
    )
    

    
    distances = np.linalg.norm(filtered_lidar, axis=1)
    lidar_resolution = 360 / len(distances)
    index_90 = int(round(90 / lidar_resolution))
    index_270 = int(round(270 / lidar_resolution))
    distance_90 = discretize(distances[index_90])
    distance_270 = discretize(distances[index_270])
    distance_difference = discretize(distance_270 - distance_90)
    
    # Discretizar velocidad para mejor aprendizaje
    discretized_speed = discretize(car_speed, step=0.1, max_value=20)

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
        discretized_min_distance = discretize(min_distance_signed)
    else:
        # Si no hay distancias válidas, asignamos infinito
        discretized_min_distance = np.inf

    return (
        discretize(distance_to_target), 
        discretize(heading_error, step=0.1, max_value=np.pi), 
        discretized_speed, 
        distance_difference,
        discretized_min_distance
    )


def compute_parking_reward( lidar_data: np.ndarray, car_pose: np.ndarray, heading: float) -> float:
        lidar_length = len(lidar_data)
        lidar_resolution = 360/300
        heading_deg = np.degrees(heading)

        # index_90 = lidar_length // 4  # Índice correspondiente a 90°.
        # index_270 = (3 * lidar_length) // 4  # Índice correspondiente a 270°.
        index_90 = int(round(heading_deg / lidar_resolution))
        index_270 = int(round((heading_deg + 180) / lidar_resolution))

        # Asignar '0' a los puntos donde no hay obstáculos ([0, 0, 0]).
        lidar_data[np.all(lidar_data == [0, 0, 0], axis=1)] = float('0')

        relative_lidar = lidar_data - car_pose  # Convertir a coordenadas relativas.
        distances = np.linalg.norm(relative_lidar, axis=1)  # Calcular distancias.

        
        # Obtener las distancias en los ángulos deseados.
        distance_90 = distances[index_90]
        distance_270 = distances[index_270]
        print(f"Distancia delante: {distance_90}")
        print(f"Distancia detrás: {distance_270}")

        # Si alguno de los valores es 0inito, penalización máxima.
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

def find_closest_corners(point_cloud):
    """Finds the closest corners of two detected obstacles from LiDAR data."""

    # Remove 'inf' values and noise
    point_cloud = point_cloud[~np.isinf(point_cloud).any(axis=1)]
    
    # Determine separation threshold using the median X value
    depth_axis = 1 if np.isclose(abs(-1.57), np.pi/2, atol=0.1) else 0

    threshold_x = np.median(point_cloud[:, 1 - depth_axis])
    
    # Split points into two obstacles
    obstacle_1 = point_cloud[point_cloud[:, 0] < threshold_x]
    obstacle_2 = point_cloud[point_cloud[:, 0] >= threshold_x]
    
    def get_corners(obstacle):
        """Finds the leftmost and rightmost points among the closest to the sensor."""
        closest_points = obstacle[np.argsort(np.abs(obstacle[:, depth_axis]))[:10]]  # 10 closest in |y|
        left_corner = closest_points[np.argmin(closest_points[:, 0])]
        right_corner = closest_points[np.argmax(closest_points[:, 0])]
        return np.array([left_corner, right_corner])

    # Get corners for both obstacles
    corners_1 = get_corners(obstacle_1)
    corners_2 = get_corners(obstacle_2)

    # Find the closest pair of corners
    distances = cdist(corners_1, corners_2)
    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
    
    return corners_1[min_idx[0]], corners_2[min_idx[1]]

def find_deepest_point(point_cloud_1, punto_1, punto_2, threshold=0.25):
    """
    Finds the deepest point within the parking gap by checking the absolute Y-coordinate values.
    Restricts points along the X-axis using a threshold but allows total freedom along the Y-axis.
    
    Parameters:
        point_cloud_1: Array of the complete point cloud.
        punto_1, punto_2: Points that define the boundaries of the parking gap.
        threshold: Margin of tolerance around the X-axis boundaries (default is 0.5).
    
    Returns:
        deepest_point: Array of size 3 (x, y, z) representing the deepest point.
                      Returns None if no points are found within the X-axis range.
    """
    # Define the X-axis boundaries of the parking gap with a threshold
    x_min = min(punto_1[0], punto_2[0]) - threshold
    x_max = max(punto_1[0], punto_2[0]) + threshold

    # Filter points that are within the X-axis range (no restriction on Y-axis)
    points_in_gap = point_cloud_1[
        (point_cloud_1[:, 0] >= x_min) & (point_cloud_1[:, 0] <= x_max)
    ]

    # If no points are within the X-axis range, return None
    if len(points_in_gap) == 0:
        return None

    # Calculate the absolute Y-coordinate values
    absolute_y_values = np.abs(points_in_gap[:, 1])

    # Find the point with the highest absolute Y-coordinate
    deepest_point = points_in_gap[np.argmax(absolute_y_values)]

    return deepest_point

point_cloud_11 = np.array([[ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [24.        , 96.14922371,  1.00111111],
       [24.        , 96.31401051,  1.00109396],
       [24.        , 96.47385586,  1.00107779],
       [24.        , 96.62911285,  1.00106255],
       [24.        , 96.78010679,  1.0010482 ],
       [24.        , 96.92713805,  1.00103467],
       [24.        , 97.07048468,  1.00102193],
       [24.        , 97.21040463,  1.00100993],
       [24.        , 97.3471378 ,  1.00099865],
       [24.        , 97.48090781,  1.00098805],
       [23.83661951, 97.535     ,  1.00100961],
       [23.53897668, 97.535     ,  1.00105684],
       [23.21196813, 97.535     ,  1.00110922],
       [22.85068924, 97.535     ,  1.0011676 ],
       [22.44908322, 97.535     ,  1.00123303],
       [21.99958135, 97.535     ,  1.00130684],
       [21.49259942, 97.535     ,  1.00139071],
       [20.91581867, 97.535     ,  1.00148676],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [35.71537181, 97.535     ,  1.00123697],
       [35.31127307, 97.535     ,  1.0011711 ],
       [34.94788554, 97.535     ,  1.00111235],
       [34.61907756, 97.535     ,  1.00105966],
       [34.31988705, 97.535     ,  1.00101216],
       [34.04625518, 97.535     ,  1.00096914],
       [33.79482962, 97.535     ,  1.00093002],
       [33.56281709, 97.535     ,  1.00089431],
       [33.34787131, 97.535     ,  1.00086161],
       [33.14800692, 97.535     ,  1.00083156],
       [32.9615325 , 97.535     ,  1.00080388],
       [32.78699799, 97.535     ,  1.00077832],
       [32.623153  , 97.535     ,  1.00075465],
       [32.46891344, 97.535     ,  1.00073269],
       [32.32333467, 97.535     ,  1.00071227],
       [32.32      , 97.42868972,  1.00072317],
       [32.32      , 97.3161872 ,  1.00073551],
       [32.32      , 97.19972597,  1.00074862],
       [32.32      , 97.07898557,  1.00076254],
       [32.32      , 96.95361376,  1.00077733],
       [32.32      , 96.82322228,  1.00079307],
       [32.32      , 96.68738196,  1.00080982],
       [32.32      , 96.545617  ,  1.00082766],
       [32.32      , 96.39739831,  1.00084668],
       [32.32      , 96.24213559,  1.00086699],
       [32.32      , 96.07916811,  1.0008887 ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ]])

point_cloud_1 = np.array([[ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [24.        , 96.09205499,  1.00099626],
       [24.        , 96.25290795,  1.00097725],
       [24.        , 96.40780486,  1.00095937],
       [24.        , 96.55719912,  1.00094254],
       [24.        , 96.70150317,  1.00092668],
       [24.        , 96.84109313,  1.00091175],
       [24.        , 96.97631296,  1.00089767],
       [24.        , 97.10747796,  1.00088441],
       [24.        , 97.23487788,  1.0008719 ],
       [24.        , 97.35877965,  1.00086012],
       [24.        , 97.47942974,  1.00084902],
       [23.89256433, 97.535     ,  1.00086022],
       [23.67842626, 97.535     ,  1.00089278],
       [23.44733516, 97.535     ,  1.00092832],
       [23.19696623, 97.535     ,  1.00096726],
       [22.92455296, 97.535     ,  1.00101006],
       [22.62677673, 97.535     ,  1.00105731],
       [22.29962161, 97.535     ,  1.00110971],
       [21.9381808 , 97.535     ,  1.00116812],
       [21.53639478, 97.535     ,  1.00123358],
       [21.08669144, 97.535     ,  1.00130743],
       [20.57948228, 97.535     ,  1.00139133],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [35.77211264, 97.535     ,  1.00139642],
       [35.26130986, 97.535     ,  1.00131189],
       [34.80862937, 97.535     ,  1.00123753],
       [34.40434951, 97.535     ,  1.00117163],
       [34.04079911, 97.535     ,  1.00111285],
       [33.71184375, 97.535     ,  1.00106014],
       [33.41251915, 97.535     ,  1.00101261],
       [33.13876464, 97.535     ,  1.00096957],
       [32.88722639, 97.535     ,  1.00093043],
       [32.65510986, 97.535     ,  1.00089471],
       [32.44006774, 97.535     ,  1.00086199],
       [32.32      , 97.4867274 ,  1.00084822],
       [32.32      , 97.36628125,  1.00085928],
       [32.32      , 97.24259754,  1.00087102],
       [32.32      , 97.11543093,  1.00088348],
       [32.32      , 96.98451577,  1.0008967 ],
       [32.32      , 96.84956373,  1.00091072],
       [32.32      , 96.71026111,  1.00092561],
       [32.32      , 96.56626576,  1.0009414 ],
       [32.32      , 96.41720353,  1.00095818],
       [32.32      , 96.26266427,  1.00097599],
       [32.32      , 96.10219713,  1.00099493],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ]])


point_cloud_2 = np.array(([ 28.17, 100.  ,   1.  ],
       [ 28.17, 100.  ,   1.  ],
       [ 28.17, 100.  ,   1.  ],
))

point_cloud_22 = np.array(([29.07018483, 99.9990929 ,  1.        ],
       [29.07018483, 99.9990929 ,  1.        ],
))


# point_cloud_3 = np.array()

# filtered_points = point_cloud_1[
#     (np.sqrt(point_cloud_1[:, 0]**2 + point_cloud_1[:, 1]**2) <= 30) &  # Radio en XY <= 30 m
#     (point_cloud_1[:, 2] >= 1) & (point_cloud_1[:, 2] <= 1.5)  &         # Z entre 1 y 1.5 m
#     ~((point_cloud_1[:, 0] == 0) & (point_cloud_1[:, 1] == 0) & (point_cloud_1[:, 2] == 0))  # Elimina puntos en (0, 0, 0)
# ]

# compute_parking_reward(point_cloud_1, point_cloud_2[0], 1.57)
# print_distance(point_cloud_1, point_cloud_2[0], 1.57)
#[ (point_cloud_1[:, 2] < 1) & (point_cloud_1[:, 2] >= 0.0)]
np.set_printoptions(suppress=True, precision=10)

def print_point_clouds(point_cloud):
    for i, vector in enumerate(point_cloud):
        # Formato del vector
        formatted_vector = f"[{vector[0]}, {vector[1]}, 0]"
        # Imprimir el vector con la coma y el salto de línea
        print(formatted_vector, end=',\n' if i < len(point_cloud) - 1 else '\n')

# print("Point Cloud 1:")
# print_point_clouds(point_cloud_1)
# print("-------------------------")
# print("Point Cloud 2:")
# print_point_clouds(point_cloud_3)
point_cloud_1 = filtrate_lidar(point_cloud_11, point_cloud_22[0], 1.57)

# point_cloud_1 = point_cloud_1[~np.all(point_cloud_1 == [0, 0, 0], axis=1)] - point_cloud_2[0]


punto_1, punto_2 = find_closest_corners(point_cloud_1)
punto_profundo = find_deepest_point(point_cloud_1, punto_1, punto_2)
n = 1
if punto_1[1] < 0:
    n = -1
punto_medio = [(punto_1[0] + punto_2[0])/2, punto_1[1]+(0.7*n), punto_1[2]]
state = get_state(point_cloud_1, punto_medio, 1.57, 1.57, 1)
print(state)
# print(filtrated_lidar)
# print_point_clouds(point_cloud_1)
x = point_cloud_1[:, 0]
y = point_cloud_1[:, 1]
z = point_cloud_1[:, 2]

# # x2 = point_cloud_2[:, 0]
# # y2 = point_cloud_2[:, 1]
# # z2 = point_cloud_2[:, 2]

# # x3 = point_cloud_3[:, 0]
# # y3 = point_cloud_3[:, 1]
# # z3 = point_cloud_3[:, 2]
# # Crear una figura y un eje 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficar los puntos
ax.scatter(x, y, z, c='b', marker='o', s=10)
# ax.scatter(x2, y2, z2, c='r', marker='o', s=10)
# ax.scatter(x3, y3, z3, c='g', marker='o', s=10)

# Graficar los puntos de referencia
ax.scatter(punto_1[0], punto_1[1], punto_1[2], c='r', marker='x', s=100, label='Punto más cercano (Obstáculo 1)')
ax.scatter(punto_2[0], punto_2[1], punto_2[2], c='g', marker='x', s=100, label='Punto más cercano (Obstáculo 2)')


print(punto_medio + point_cloud_22[0])
ax.scatter(punto_medio[0], punto_medio[1], punto_medio[2], c='b', marker='x', s=100, label='Punto más medio')
ax.scatter(punto_profundo[0], punto_profundo[1], punto_profundo[2], c='g', marker='x', s=100, label='Punto más profundo')
ax.scatter(28.16-29.07018483, 96.8-100, punto_profundo[2], c='r', marker='x', s=100, label='Punto más profundo')

ax.set_xlim([-10, 10])  # Límite del eje x
ax.set_ylim([-10, 10])  # Límite del eje y
ax.set_zlim([0, 1.5])    # Límite del eje z
# Etiquetas y título
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Nube de Puntos Lidar')

# Mostrar el gráfico
plt.show()

