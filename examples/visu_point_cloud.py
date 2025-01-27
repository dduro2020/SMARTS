import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
    # Asignar '0' a los puntos inválidos (donde todo es [0, 0, 0])
    lidar_data[np.all(lidar_data == [0, 0, 0], axis=1)] = float('0')

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


def print_distance(lidar_data: np.ndarray, car_pose: np.ndarray, heading: float):
        filtrate_l = filtrate_lidar(lidar_data, car_pose, heading)
        print(filtrate_l)
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


point_cloud_1 = np.array([[ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [ 32.32      , 100.70728057,   1.00073648],
       [ 32.32      , 100.61795706,   1.00073402],
       [ 32.32      , 100.5291876 ,   1.00073191],
       [ 32.32      , 100.44088988,   1.00073012],
       [ 32.32      , 100.35298326,   1.00072867],
       [ 32.32      , 100.26538851,   1.00072753],
       [ 32.32      , 100.17802752,   1.00072672],
       [ 32.32      , 100.09082298,   1.00072623],
       [ 32.32      , 100.00369816,   1.00072606],
       [ 32.32      ,  99.91657658,   1.0007262 ],
       [ 32.32      ,  99.82938179,   1.00072667],
       [ 32.32      ,  99.74203707,   1.00072745],
       [ 32.32      ,  99.6544652 ,   1.00072856],
       [ 32.32      ,  99.56658812,   1.00072999],
       [ 32.32      ,  99.47832671,   1.00073174],
       [ 32.32      ,  99.38960048,   1.00073383],
       [ 32.32      ,  99.30032726,   1.00073625],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [  0.        ,   0.        ,   0.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ],
       [ 28.16      , 100.        ,   1.        ]])


point_cloud_2 = np.array(([ 28.17, 100.  ,   1.  ],
       [ 28.17, 100.  ,   1.  ],
       [ 28.17, 100.  ,   1.  ],
))


# point_cloud_3 = np.array()

# filtered_points = point_cloud_1[
#     (np.sqrt(point_cloud_1[:, 0]**2 + point_cloud_1[:, 1]**2) <= 30) &  # Radio en XY <= 30 m
#     (point_cloud_1[:, 2] >= 1) & (point_cloud_1[:, 2] <= 1.5)  &         # Z entre 1 y 1.5 m
#     ~((point_cloud_1[:, 0] == 0) & (point_cloud_1[:, 1] == 0) & (point_cloud_1[:, 2] == 0))  # Elimina puntos en (0, 0, 0)
# ]

compute_parking_reward(point_cloud_1, point_cloud_2[0], 1.57)
print_distance(point_cloud_1, point_cloud_2[0], 1.57)
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

point_cloud_1 = point_cloud_1[~np.all(point_cloud_1 == [0, 0, 0], axis=1)] - point_cloud_2[0]
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

