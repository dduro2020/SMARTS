import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np

# Define la red DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Inicializa el entorno y la red
env = gym.make("MountainCar-v0")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = DQN(state_size, action_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
memory = deque(maxlen=10000)
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# Función para seleccionar una acción
def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_size - 1)
    else:
        with torch.no_grad():
            return torch.argmax(model(torch.FloatTensor(state))).item()

# Función para entrenar la red
def train_model():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    
    # Convertir la lista de arrays de NumPy en un solo array de NumPy
    states = np.array([transition[0] for transition in batch])
    actions = np.array([transition[1] for transition in batch])
    rewards = np.array([transition[2] for transition in batch])
    next_states = np.array([transition[3] for transition in batch])
    dones = np.array([transition[4] for transition in batch])

    # Convertir los arrays de NumPy en tensores de PyTorch
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    # Calcular los valores Q actuales y los objetivos
    current_q = model(states).gather(1, actions.unsqueeze(1))
    next_q = model(next_states).max(1)[0].detach()
    target_q = rewards + gamma * next_q * (1 - dones)

    # Calcular la pérdida y actualizar la red
    loss = nn.MSELoss()(current_q.squeeze(), target_q)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# Entrenamiento
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        train_model()
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")