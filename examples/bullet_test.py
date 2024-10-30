
import pybullet as p  # Importar PyBullet

# Inicia el simulador en modo GUI
p.connect(p.GUI)
p.resetSimulation()

# Carga una escena básica
try:
    # Asegúrate de que los archivos URDF estén en el directorio actual o en la ruta de pybullet
    p.setAdditionalSearchPath(".")  # Directorio actual
    # p.loadURDF("plane.urdf")
    # cube_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5])  # Añadir cubo

    # Prueba de rayos en PyBullet estándar
    ray_from = [0, 0, 1]
    ray_to = [0, 0, 0]
    result = p.rayTest(ray_from, ray_to)

    print("Ray test result in standard pybullet:", result)
except Exception as e:
    print("Error loading URDF files or running ray test:", e)
finally:
    # Desconectar después de la ejecución
    p.disconnect()
