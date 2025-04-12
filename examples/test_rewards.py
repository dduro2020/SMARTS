import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Configuración inicial
fig = plt.figure(figsize=(12, 6))
ax_plot = fig.add_subplot(121)
ax_controls = fig.add_subplot(122)
ax_controls.axis('off')  # Ocultar ejes del panel de control

# Valores iniciales
dist_h_init = 5.0
dist_v_init = 5.0
orient_init = 0

# Configuración de la gráfica
ax_plot.set_xlim(0, 10)
ax_plot.set_ylim(-10, 10)
ax_plot.set_xlabel('Iteraciones')
ax_plot.set_ylabel('Valor de la ecuación')
ax_plot.set_title('Historial de valores')

# Variables para almacenar el historial
history = []
current_iteration = 0 

# Función de ejemplo (puedes cambiarla por tu ecuación)
def ecuacion(dist_h, dist_v, orient_diff):
    speed = 0.5
    dist_to_target = np.linalg.norm([dist_h, dist_v])

    # Escalar la distancia al objetivo a un rango manejable
    if dist_to_target <= 0.25:
        dist_to_target = 0.2
    if abs(dist_v) <= 0.2 and abs(dist_h) <= 0.1:
        print("MUY CERCAAA")
        dist_to_target = 0.1
    
    if abs(dist_h) <= 0.1:
        dist_h = 0.1
        
    a = 3.5  # Ajuste de escala
    b = 2.5  # Controla la velocidad de caída exponencial
    c = -0.75  # Límite inferior de penalización

    orientation_reward = a * np.exp(-b * orient_diff) + c
    orientation_reward = max(-0.5, min(orientation_reward, 3))
    
    # Recompensa por distancia (escalada a [-1, 1])
    distance_reward = 1 / (1 + np.exp(3 * (dist_to_target - 0.4)))
    if dist_to_target > 7.5 or dist_h < -0.75: #emula obstaculo horizontal
        distance_reward = -5  # Penalización máxima por distancia


    # if abs(dist_h) < 0.3:
    #     orientation_reward = -(((5 * np.pi) / 12) * orient_diff) + (0.1/dist_h)
    #     orientation_reward = max(-0.5, min(orientation_reward, 1))  # Asegurar rango [-0.5, 1]
    # else:
    #     orientation_reward = 0

    if orient_diff > ((5 * np.pi) / 12):  # 75º
        orientation_reward = -5 # Penalización máxima por orientación

    # 3. Penalización por velocidad (escalada a [-1, 0])
    if abs(speed) > 2:
        speed_penalty = -0.5  # Penalización máxima por velocidad
    elif abs(speed) < 0.5 and dist_to_target < 0.2:
        speed_penalty = distance_reward/3
    else:
        speed_penalty = 0

    # 4. Bonificación por detenerse correctamente (escalada a [0, 1])
    if orient_diff < 0.1 and abs(dist_h) < 0.2 and abs(dist_v) < 0.25:
        stopping_bonus = 20# Bonificación máxima por detenerse
        print("CONSEGUIDO!!")
    else:
        stopping_bonus = 0

    # 6. Cálculo final de la recompensa (escalada a [-1, 1])
    reward = (
        distance_reward
        + orientation_reward
        + speed_penalty
        + stopping_bonus
    )
    return reward*100

# Creación de sliders
ax_dist_h = plt.axes([0.65, 0.7, 0.25, 0.03])
ax_dist_v = plt.axes([0.65, 0.6, 0.25, 0.03])
ax_orient = plt.axes([0.65, 0.5, 0.25, 0.03])

slider_dist_h = Slider(ax_dist_h, 'Dist. Horizontal', -6, 6.0, valinit=dist_h_init)
slider_dist_v = Slider(ax_dist_v, 'Dist. Vertical', -6, 6.0, valinit=dist_v_init)
slider_orient = Slider(ax_orient, 'Orientación', -np.pi/2, np.pi/2, valinit=orient_init)

exit_ax = plt.axes([0.65, 0.3, 0.25, 0.05])
exit_button = Button(exit_ax, 'Salir (X)', color='lightgoldenrodyellow')

# Inicializar gráfica
line, = ax_plot.plot([], [], 'r-', marker='o', markersize=5)  # Línea con puntos

def update(val):
    global current_iteration, history
    
    try:
        # Obtener valores actuales
        dist_h = slider_dist_h.val
        dist_v = slider_dist_v.val
        angle = slider_orient.val
        
        # Calcular nuevo valor (aseguramos que es float)
        new_value = float(ecuacion(dist_h, dist_v, angle))
        history.append(new_value)
        current_iteration += 1
        
        # Actualizar datos de la gráfica
        x_data = list(range(1, len(history)+1))
        line.set_data(x_data, history)
        
        # Ajustar límites de los ejes con valores por defecto si history está vacío
        ax_plot.set_xlim(0.5, max(10, len(history)+0.5))
        
        if len(history) > 0:
            y_min = min(history) - 1
            y_max = max(history) + 1
            ax_plot.set_ylim(y_min, y_max)
        else:
            ax_plot.set_ylim(-10, 10)
            
        fig.canvas.draw_idle()
    except Exception as e:
        print(f"Error en actualización: {e}")
        ax_plot.set_ylim(-10, 10)

# Configurar eventos
slider_dist_h.on_changed(update)
slider_dist_v.on_changed(update)
slider_orient.on_changed(update)

# Función para salir
def exit_program(event):
    plt.close()

exit_button.on_clicked(exit_program)

# Llamada inicial para mostrar el primer punto
update(None)

# Mostrar gráfica
plt.tight_layout()
plt.show()