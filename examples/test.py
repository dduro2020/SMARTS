# import matplotlib.pyplot as plt

# epsilon_0 = 0.99
# min_epsilon = 0.01
# decay_rate = 0.985
# episodes = 300

# epsilon_values = []
# epsilon = epsilon_0

# for episode in range(episodes):
#     epsilon = max(min_epsilon, epsilon * decay_rate)
#     epsilon_values.append(epsilon)

# plt.plot(epsilon_values)
# plt.xlabel('Episodio')
# plt.ylabel('Epsilon')
# plt.title('Evolucion epsilon')
# plt.show()


import matplotlib.pyplot as plt
import re

# Ruta del archivo
file_path = "/home/duro/SMARTS/examples/data.log"

# Inicializamos listas para almacenar los datos
episodes = []
scores = []

# Leer el archivo línea por línea
with open(file_path, "r") as file:
    for line in file:
        # Buscar episodios y puntajes usando regex
        episode_match = re.search(r"│\s+(\d+)/\d+\s+│", line)
        score_match = re.search(r"│\s+([\d\.]+)\s+-\s+SingleAgent\s+│", line)

        if episode_match and score_match:
            episode = int(episode_match.group(1))
            score = float(score_match.group(1))

            # Almacenar los datos en las listas
            episodes.append(episode)
            scores.append(score)

# Crear la gráfica
plt.figure(figsize=(10, 6))
plt.plot(episodes, scores, marker="o", linestyle="-", color="b")
plt.title("Episodios vs Scores")
plt.xlabel("Episodios")
plt.ylabel("Scores")
plt.grid(True)
plt.show()

