import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

def plot_training_data_live(log_file="/home/duro/SMARTS/examples/training_log.csv", update_interval=5, max_episodes=3500):
    """Carga y grafica los datos del entrenamiento en tiempo real."""
    sns.set_theme(style="darkgrid")

    # Crear la figura y los ejes antes del bucle
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    while True:
        try:
            # Cargar datos
            df = pd.read_csv(log_file)

            # Verificar si el DataFrame está vacío
            if df.empty:
                print("El archivo CSV no tiene datos todavía. Esperando...")
                time.sleep(update_interval)
                continue  # Salta la iteración y vuelve a intentar

            # Verificar si ya se alcanzó el número máximo de Episodes
            if df["episode"].iloc[-1] >= max_episodes:
                print(f"Se alcanzaron {max_episodes} Episodes. Finalizando...")
                break

            # Limpiar los ejes antes de actualizar
            for ax in axes.flat:
                ax.clear()

            # Gráfico de recompensa por Episode
            sns.lineplot(ax=axes[0, 0], x=df["episode"], y=df["reward"])
            axes[0, 0].set_title("Episode Reward")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")

            # Gráfico de pérdida media por Episode
            sns.lineplot(ax=axes[0, 1], x=df["episode"], y=df["loss"])
            axes[0, 1].set_title("Episode MLoss")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("MLoss")

            # Gráfico de epsilon
            sns.lineplot(ax=axes[1, 0], x=df["episode"], y=df["epsilon"])
            axes[1, 0].set_title("Episode Epsilon")
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Epsilon")

            # Gráfico de distancia al objetivo
            sns.lineplot(ax=axes[1, 1], x=df["episode"], y=df["distance_to_target"])
            axes[1, 1].set_title("Target MDistance")
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("MDistance")

            # Ajustar layout y actualizar gráficos
            plt.tight_layout()
            plt.pause(update_interval)  # Esperar antes de la siguiente actualización

        except FileNotFoundError:
            print(f"Archivo {log_file} no encontrado. Esperando datos...")
            time.sleep(update_interval)
        except pd.errors.EmptyDataError:
            print("El archivo CSV está vacío. Esperando datos...")
            time.sleep(update_interval)
        except IndexError:
            print("El DataFrame está vacío o sin episodes válidos. Esperando datos...")
            time.sleep(update_interval)
        except KeyboardInterrupt:
            print("Interrupción del usuario (Ctrl+C). Finalizando...")
            break
        except Exception as e:
            print(f"Error inesperado: {e}")
            break

    plt.show()

# Llamar a la función para visualizar en tiempo real
plot_training_data_live()
