import matplotlib.pyplot as plt

epsilon_0 = 0.99
min_epsilon = 0.0
decay_rate = 0.99
episodes = 500

epsilon_values = []
epsilon = epsilon_0

for episode in range(episodes):
    epsilon = max(min_epsilon, epsilon * decay_rate)
    epsilon_values.append(epsilon)

plt.plot(epsilon_values)
plt.xlabel('Episodio')
plt.ylabel('Epsilon')
plt.title('Evolucion epsilon')
plt.show()


# import matplotlib.pyplot as plt
# import re

# # Ruta del archivo
# file_path = "/home/duro/SMARTS/examples/data.log"

# # Inicializamos listas para almacenar los datos
# episodes = []
# scores = []
# steps = []

# # Leer el archivo línea por línea
# with open(file_path, "r") as file:
#     for line in file:
#         episode_match = re.search(r"\u2502\s+(\d+)/\d+\s+\u2502", line)  # Episodio
#         score_match = re.search(r"\u2502\s+([\d\.]+)\s+-\s+SingleAgent\s+\u2502", line)  # Score
#         steps_match = re.search(r"\u2502\s+(\d+)\s+\u2502", line)  # Total Steps

#         if episode_match and score_match and steps_match:
#             episode = int(episode_match.group(1))
#             score = float(score_match.group(1))
#             step = int(steps_match.group(1))

#             # Almacenar los datos en las listas
#             episodes.append(episode)
#             scores.append(score)
#             steps.append(step)

# # Crear la primera gráfica: Episodes vs Scores
# plt.figure(figsize=(12, 6))
# plt.subplot(2, 1, 1)  # Subgráfica 1
# plt.plot(episodes, scores, marker="o", linestyle="--", color="b")
# plt.title("Episodes vs Scores")
# plt.xlabel("Episodes")
# plt.ylabel("Scores")
# plt.grid(True)

# # Crear la segunda gráfica: Episodes vs Total Steps
# plt.subplot(2, 1, 2)  # Subgráfica 2
# plt.plot(episodes, steps, marker="s", linestyle="--", color="g")
# plt.title("Episodes vs Total Steps")
# plt.xlabel("Episodes")
# plt.ylabel("Total Steps")
# plt.grid(True)

# # Ajustar el layout y mostrar las gráficas
# plt.tight_layout()
# plt.show()



# import matplotlib.pyplot as plt

# # Leer las distancias desde el archivo distances.log
# file_path = "/home/duro/SMARTS/examples/distances.log"

# # Leer las distancias del archivo
# with open(file_path, "r") as file:
#     distances = [float(line.strip()) for line in file if line.strip()]

# # Crear una lista de pasos (steps) para el eje x
# steps = list(range(len(distances)))

# # Crear la gráfica
# plt.figure(figsize=(10, 6))
# plt.plot(steps, distances, marker="o", linestyle="-", color="b")
# plt.title("dist diff vs steps")
# plt.xlabel("steps")
# plt.ylabel("dist diff")
# plt.grid(True)
# plt.show()

# import matplotlib.pyplot as plt

# # Leer y procesar el archivo
# file_path = "/home/duro/SMARTS/examples/distances.log"  # Cambia esto por la ruta de tu archivo
# data = {}

# with open(file_path, "r") as file:
#     current_key = None
#     for line in file:
#         line = line.strip()
#         if line.startswith("dist_"):  # Detectar nombres de bloques
#             current_key = line
#             data[current_key] = []
#         elif line and current_key:  # Agregar distancias al bloque actual
#             data[current_key].append(float(line))

# # Graficar los datos
# fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Crear 4 subgráficas en una cuadrícula 2x2
# fig.suptitle("Distance difference evolution")

# # Asegurarse de que no hay más de 4 bloques
# keys = list(data.keys())[:4]  # Seleccionar los primeros 4 bloques si hay más
# for ax, key in zip(axs.flat, keys):
#     ax.plot(data[key], marker='o')
#     ax.set_title(key)
#     ax.set_xlabel("Steps")
#     ax.set_ylabel("Distance difference")
#     ax.set_ylim(-4, 4)

# # Ajustar el diseño
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reservar espacio para el título principal
# plt.show()