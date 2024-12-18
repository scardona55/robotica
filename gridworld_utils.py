import json
import random

# Clase para definir el entorno de la grilla con obstáculos
class GridWorld:
    def __init__(self, width, height, start, goal, obstacles):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        new_x, new_y = x, y

        if action == 'up':
            new_y = max(y - 1, 0)
        elif action == 'down':
            new_y = min(y + 1, self.height - 1)
        elif action == 'left':
            new_x = max(x - 1, 0)
        elif action == 'right':
            new_x = min(x + 1, self.width - 1)

        proposed_state = (new_x, new_y)

        if proposed_state in self.obstacles:
            reward = -0.5
            done = False
            return self.state, reward, done
        else:
            self.state = proposed_state
            if self.state == self.goal:
                reward = 1
                done = True
            else:
                reward = -0.01
                done = False
            return self.state, reward, done

# Función para inicializar los parámetros del entorno
def initialize_environment(width,height,goal,obstacles):
    start = (0, 0)
    assert start not in obstacles, "La posición de inicio no puede ser un obstáculo."
    assert goal not in obstacles, "La posición de meta no puede ser un obstáculo."

    return GridWorld(width, height, start, goal, obstacles)

# Función para inicializar la tabla Q
def initialize_q_table(width, height, obstacles, actions):
    Q = {}
    for x in range(width):
        for y in range(height):
            if (x, y) in obstacles:
                continue
            Q[(x, y)] = {action: 0 for action in actions}
    return Q

# Función para elegir una acción usando epsilon-greedy
def choose_action(state, Q, actions, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    else:
        return max(Q[state], key=Q[state].get)

# Función para entrenar al agente
def train_agent(env, Q, episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = choose_action(state, Q, env.actions, epsilon)
            next_state, reward, done = env.step(action)

            if next_state in env.obstacles:
                continue

            if next_state not in Q:
                continue

            best_next_action = max(Q[next_state], key=Q[next_state].get)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            state = next_state

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return Q

# Función para extraer la política de la tabla Q
def extract_policy(Q):
    policy = {}
    for state in Q:
        best_action = max(Q[state], key=Q[state].get)
        policy[str(state)] = best_action
    return policy

# Función para exportar la política a un archivo JSON
def export_policy(policy, filename='policy.json'):
    with open(filename, 'w') as f:
        json.dump(policy, f, indent=4)

# Función para visualizar la política
def visualize_policy(policy, width, height, obstacles, goal):
    print("\nPolítica aprendida:")
    for y in range(height):
        row = ""
        for x in range(width):
            pos = (x, y)
            if pos in obstacles:
                row += "X "
            elif pos == goal:
                row += "G "
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
                    row += "? "
        print(row)
