import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Recompensa simplificada
def compute_reward(horizontal_dist, car_orient):
    target_orient = 0.0
    orient_diff = np.abs(np.arctan2(np.sin(car_orient - target_orient), np.cos(car_orient - target_orient)))

    orient_weight = 1 / (1 + np.exp(10 * (horizontal_dist - 0.5)))
    distance_reward = 1 / (1 + np.exp(0.8 * (horizontal_dist - 3)))
    orientation_reward = orient_weight * (3.5 * np.exp(-2.5 * orient_diff) - 0.75)
    
    reward = distance_reward + np.clip(orientation_reward, -0.5, 2)

    # Bonus si se cumplen las condiciones en la malla
    bonus_mask = (orient_diff < 0.1) & (horizontal_dist < 0.25)
    reward += bonus_mask * 10

    return reward



# Rango más pequeño
horizontal_vals = np.linspace(-1.0, 6.0, 100)
orient_vals = np.linspace(-1.5, 1.5, 100)
H, O = np.meshgrid(horizontal_vals, orient_vals)

# Calcular recompensa
Z = np.vectorize(compute_reward)(np.abs(H), O)

# Plot 3D simplificado
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(H, O, Z, cmap='viridis', rstride=3, cstride=3)

ax.set_xlabel('Distancia horizontal (Y)')
ax.set_ylabel('Orientación (rad)')
ax.set_zlabel('Recompensa')
ax.set_title('Recompensa según orientación y posición lateral')

plt.tight_layout()
plt.show()
