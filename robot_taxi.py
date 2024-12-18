import gym
import numpy as np
import random
from IPython.display import clear_output
import time
import json

# Definir el entorno de la grilla con obstáculos
class GridWorld:
    def __init__(self, width, height, start, goal, obstacles):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles  # Lista de posiciones de obstáculos
        self.state = start
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        new_x, new_y = x, y  # Variables para la nueva posición

        if action == 'up':
            new_y = max(y - 1, 0)
        elif action == 'down':
            new_y = min(y + 1, self.height - 1)
        elif action == 'left':
            new_x = max(x - 1, 0)
        elif action == 'right':
            new_x = min(x + 1, self.width - 1)

        proposed_state = (new_x, new_y)

        # Verificar si la nueva posición es un obstáculo
        if proposed_state in self.obstacles:
            # Penalización por intentar moverse hacia un obstáculo
            reward = -0.5
            done = False
            # El robot no cambia de posición
            return self.state, reward, done
        else:
            self.state = proposed_state
            if self.state == self.goal:
                reward = 1
                done = True
            else:
                reward = -0.01  # Pequeña penalización por cada movimiento
                done = False
            return self.state, reward, done

"""##Parametros del mundo"""

# Parámetros del entorno
width = 4
height = 4
start = (0, 0)
goal = (3, 3)
obstacles = [(1, 1),]  # Definir posiciones de obstáculos

# Asegurarse de que las posiciones de inicio y meta no estén entre los obstáculos
assert start not in obstacles, "La posición de inicio no puede ser un obstáculo."
assert goal not in obstacles, "La posición de meta no puede ser un obstáculo."

env = GridWorld(width, height, start, goal, obstacles)

"""##parametros del q-learning"""

# Parámetros de Q-learning
alpha = 0.1      # Tasa de aprendizaje
gamma = 0.99     # Factor de descuento
epsilon = 1.0    # Tasa de exploración inicial
epsilon_min = 0.1
epsilon_decay = 0.995
episodes = 1000

"""##Desarrollo de la política"""

# Inicializar la tabla Q
Q = {}
for x in range(width):
    for y in range(height):
        if (x, y) in obstacles:
            continue  # No inicializar Q para obstáculos
        Q[(x, y)] = {action: 0 for action in env.actions}

# Función para elegir una acción usando epsilon-greedy
def choose_action(state):
    if random.uniform(0,1) < epsilon:
        return random.choice(env.actions)
    else:
        return max(Q[state], key=Q[state].get)

# Entrenamiento del agente
for episode in range(episodes):
    state = env.reset()
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done = env.step(action)

        if next_state in obstacles:
            # Si el próximo estado es un obstáculo, no actualizamos Q
            continue

        if next_state not in Q:
            # Si el próximo estado no está en Q (es un obstáculo), saltamos la actualización
            continue

        # Actualizar Q-valor
        best_next_action = max(Q[next_state], key=Q[next_state].get)
        Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

        state = next_state

    # Decaer la tasa de exploración
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

# Extracción de la política
policy = {}
for state in Q:
    best_action = max(Q[state], key=Q[state].get)
    policy[str(state)] = best_action

"""##Exportar y visualizar"""

# Exportar la política a JSON
with open('policy.json', 'w') as f:
    json.dump(policy, f, indent=4)

print("Entrenamiento completado y política exportada a 'policy.json'.")

# Para ver la política en Colab, puedes imprimirla
print("\nPolítica aprendida:")
for y in range(height):
    row = ""
    for x in range(width):
        pos = (x, y)
        if pos in obstacles:
            row += "X "  # Representar obstáculos con 'X'
        elif pos == goal:
            row += "G "  # Representar la meta con 'G'
        else:
            action = policy.get(str(pos), 'N/A')
            if action == 'up':
                row += "U "
            elif action == 'down':
                row += "D "
            elif action == 'left':
                row += "L "
            elif action == 'right':
                row += "R "
            else:
                row += "? "  # Acción desconocida
    print(row)