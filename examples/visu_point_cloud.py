import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_parking_reward( lidar_data: np.ndarray, car_pose: np.ndarray) -> float:
        heading = 1.57
        lidar_length = len(lidar_data)
        lidar_resolution = 360/300
        heading_deg = np.degrees(heading)

        # index_90 = lidar_length // 4  # Índice correspondiente a 90°.
        # index_270 = (3 * lidar_length) // 4  # Índice correspondiente a 270°.
        index_90 = int(round(heading_deg / lidar_resolution))
        index_270 = int(round((heading_deg + 180) / lidar_resolution))

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


point_cloud_1 = np.array([[  0.        ,   0.        ,   0.        ],
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
       [ 24.        ,  99.32602977,   1.0006269 ],
       [ 24.        ,  99.40230972,   1.00062454],
       [ 24.        ,  99.47805031,   1.00062247],
       [ 24.        ,  99.553323  ,   1.00062068],
       [ 24.        ,  99.62819758,   1.00061918],
       [ 24.        ,  99.70274238,   1.00061795],
       [ 24.        ,  99.77702457,   1.00061699],
       [ 24.        ,  99.85111037,   1.00061631],
       [ 24.        ,  99.92506534,   1.0006159 ],
       [ 24.        ,  99.99895454,   1.00061577],
       [ 24.        , 100.07284282,   1.0006159 ],
       [ 24.        , 100.14679503,   1.0006163 ],
       [ 24.        , 100.22087623,   1.00061697],
       [ 24.        , 100.29515195,   1.00061792],
       [ 24.        , 100.3696884 ,   1.00061914],
       [ 24.        , 100.44455271,   1.00062063],
       [ 24.        , 100.51981319,   1.00062241],
       [ 24.        , 100.59553956,   1.00062448],
       [ 24.        , 100.67180325,   1.00062683],
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
       [ 32.32      , 100.71182754,   1.00084553],
       [ 32.32      , 100.6095736 ,   1.00084309],
       [ 32.32      , 100.50786305,   1.00084103],
       [ 32.32      , 100.40660302,   1.00083935],
       [ 32.32      , 100.30570223,   1.00083805],
       [ 32.32      , 100.20507071,   1.00083711],
       [ 32.32      , 100.10461941,   1.00083655],
       [ 32.32      , 100.00425993,   1.00083635],
       [ 32.32      ,  99.90390418,   1.00083652],
       [ 32.32      ,  99.8034641 ,   1.00083705],
       [ 32.32      ,  99.70285133,   1.00083795],
       [ 32.32      ,  99.60197689,   1.00083923],
       [ 32.32      ,  99.50075088,   1.00084088],
       [ 32.32      ,  99.39908217,   1.0008429 ],
       [ 32.32      ,  99.29687802,   1.0008453 ],
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
       [  0.        ,   0.        ,   0.        ]])


point_cloud_2 = np.array([[ 27.52807709, 100.        ,   1.        ],
       [ 27.52807709, 100.        ,   1.        ]])


# point_cloud_3 = np.array()

# filtered_points = point_cloud_1[
#     (np.sqrt(point_cloud_1[:, 0]**2 + point_cloud_1[:, 1]**2) <= 30) &  # Radio en XY <= 30 m
#     (point_cloud_1[:, 2] >= 1) & (point_cloud_1[:, 2] <= 1.5)  &         # Z entre 1 y 1.5 m
#     ~((point_cloud_1[:, 0] == 0) & (point_cloud_1[:, 1] == 0) & (point_cloud_1[:, 2] == 0))  # Elimina puntos en (0, 0, 0)
# ]

compute_parking_reward(point_cloud_1, point_cloud_2[0])

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

x = point_cloud_1[:, 0]
y = point_cloud_1[:, 1]
z = point_cloud_1[:, 2]

# x2 = point_cloud_2[:, 0]
# y2 = point_cloud_2[:, 1]
# z2 = point_cloud_2[:, 2]

# x3 = point_cloud_3[:, 0]
# y3 = point_cloud_3[:, 1]
# z3 = point_cloud_3[:, 2]
# Crear una figura y un eje 3D
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

