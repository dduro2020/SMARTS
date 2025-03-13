import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque

# Definir la arquitectura de la red neuronal para DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Clase para almacenar y muestrear experiencias (Experience Replay)
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Clase del agente DQN
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Parámetros del DQN
        self.gamma = 0.99  # Factor de descuento
        self.epsilon = 1.0  # Exploración inicial
        self.epsilon_min = 0.01  # Mínimo valor de epsilon
        self.epsilon_decay = 0.99  # Tasa de decaimiento de epsilon
        self.learning_rate = 3e-3  # Tasa de aprendizaje
        self.batch_size = 512  # Tamaño del lote para entrenamiento
        self.memory = ReplayBuffer(10000)  # Buffer de experiencias
        
        # Redes neuronal principal y objetivo
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
    
    def select_action(self, state):
        # Estrategia epsilon-greedy
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state)
        return q_values.argmax().item()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Muestrear un lote de experiencias
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Extraer componentes del lote
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(np.array(batch[1])).view(-1, 1).to(self.device)
        rewards = torch.FloatTensor(np.array(batch[2])).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(np.array(batch[4])).view(-1, 1).to(self.device)
        
        # Calcular valores Q actuales
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Calcular valores Q esperados
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Calcular la pérdida (usando Huber loss para mayor estabilidad)
        loss = F.smooth_l1_loss(current_q_values, expected_q_values)
        
        # Optimizar el modelo
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filename):
        torch.save(self.policy_net.state_dict(), filename)
    
    def load(self, filename):
        self.policy_net.load_state_dict(torch.load(filename))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def run_dqn(episodes, is_training=True, render=False):
    env = gym.make('MountainCar-v0', render_mode='human' if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    # Cargar modelo si no estamos entrenando
    if not is_training:
        try:
            agent.load('/home/duro/SMARTS/examples/mountain_car_dqn.pth')
            agent.epsilon = 0  # Sin exploración durante la evaluación
        except:
            print("No se encontró el modelo guardado. Usando un modelo sin entrenar.")
    
    rewards_per_episode = np.zeros(episodes)
    
    # Frecuencia de actualización de la red objetivo
    target_update_frequency = 10
    
    for i in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        # Modificación de la recompensa para hacer el aprendizaje más eficiente
        position_max = -1.2  # Seguimiento de la posición máxima alcanzada
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Recompensa modificada para fomentar la exploración
            position = next_state[0]
            velocity = next_state[1]
            
            if is_training:
                agent.memory.push(state, action, reward, next_state, done)
                agent.learn()
            
            state = next_state
            episode_reward += reward  # Usar la recompensa original para registro
        
        rewards_per_episode[i] = episode_reward
        # Actualizar epsilon (exploración)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        
        # Actualizar la red objetivo periódicamente
        if is_training and (i + 1) % target_update_frequency == 0:
            agent.update_target_network()
        
        # Imprimir progreso
        # if (i + 1) % 10 == 0:
        print(f'Episodio {i+1}/{episodes}, Recompensa: {episode_reward}, Epsilon: {agent.epsilon:.4f}')
    
    env.close()
    
    # Guardar el modelo entrenado
    if is_training:
        agent.save('/home/duro/SMARTS/examples/mountain_car_dqn.pth')
        
    # Visualizar resultados
    mean_rewards = np.zeros(episodes)
    for t in range(episodes):
        mean_rewards[t] = np.mean(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.figure(figsize=(10, 6))
    plt.plot(mean_rewards)
    plt.title('Recompensa promedio por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Recompensa promedio')
    plt.grid(True)
    plt.savefig('/home/duro/SMARTS/examples/mountain_car_dqn.png')
    plt.show()
    
    return agent

if __name__ == '__main__':
    # Entrenar el agente DQN
    # run_dqn(500, is_training=True, render=False)
    
    # Evaluar el agente entrenado
    run_dqn(10, is_training=False, render=True)