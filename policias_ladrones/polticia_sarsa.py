# Importaciones necesarias
import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pandas as pd
import json
from google.colab import files

# Configuración para reproducibilidad
random.seed(42)
np.random.seed(42)

# Definición del Entorno Personalizado
class PoliceThiefGridEnv(gym.Env):
    """
    Entorno personalizado para el problema de policías y ladrones en una grilla.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_size=(5,5), obstacles=[]):
        super(PoliceThiefGridEnv, self).__init__()
        
        self.grid_height, self.grid_width = grid_size
        self.obstacles = obstacles
        
        # Definir el espacio de acciones: 0=up, 1=down, 2=left, 3=right
        self.action_space = spaces.Discrete(4)
        
        # Definir el espacio de estados: posición del policía y posición del ladrón
        # Cada posición puede estar en cualquier celda libre
        self.observation_space = spaces.Tuple((
            spaces.Discrete(self.grid_height * self.grid_width),
            spaces.Discrete(self.grid_height * self.grid_width)
        ))
        
        # Inicializar posiciones
        self.reset()
    
    def reset(self):
        """
        Reinicia el entorno a un estado inicial.
        """
        # Generar posiciones aleatorias para policía y ladrón evitando obstáculos y superposición
        free_cells = [(i, j) for i in range(self.grid_height) 
                             for j in range(self.grid_width) 
                             if (i, j) not in self.obstacles]
        
        self.police_pos = random.choice(free_cells)
        free_cells.remove(self.police_pos)
        self.thief_pos = random.choice(free_cells)
        
        return self._get_state()
    
    def _get_state(self):
        """
        Retorna el estado actual.
        """
        return (self._pos_to_index(self.police_pos), self._pos_to_index(self.thief_pos))
    
    def _pos_to_index(self, pos):
        """
        Convierte una posición (i,j) a un índice único.
        """
        return pos[0] * self.grid_width + pos[1]
    
    def _index_to_pos(self, index):
        """
        Convierte un índice único a una posición (i,j).
        """
        return (index // self.grid_width, index % self.grid_width)
    
    def step(self, action, agent_type):
        """
        Ejecuta una acción para un agente específico.
        agent_type: 0 para policía, 1 para ladrón
        """
        if agent_type == 0:
            self.police_pos = self._move(self.police_pos, action)
        elif agent_type == 1:
            self.thief_pos = self._move(self.thief_pos, action)
        
        done = False
        reward = 0
        
        # Verificar captura
        if self.police_pos == self.thief_pos:
            done = True
            if agent_type == 0:
                reward = 10  # Recompensa para policía por captura
            else:
                reward = -10  # Penalización para ladrón por ser capturado
        else:
            # Recompensa mínima por cada paso para incentivar acciones eficientes
            reward = -1 if agent_type == 0 else 1
        
        return self._get_state(), reward, done, {}
    
    def _move(self, position, action):
        """
        Mueve a un agente en la dirección especificada si es posible.
        """
        i, j = position
        if action == 0 and i > 0:  # up
            new_pos = (i-1, j)
        elif action == 1 and i < self.grid_height -1:  # down
            new_pos = (i+1, j)
        elif action == 2 and j > 0:  # left
            new_pos = (i, j-1)
        elif action == 3 and j < self.grid_width -1:  # right
            new_pos = (i, j+1)
        else:
            new_pos = position  # Acción inválida, no mover
        
        # Verificar si la nueva posición es un obstáculo
        if new_pos in self.obstacles:
            return position  # No mover si es obstáculo
        else:
            return new_pos
    
    def render(self, mode='human'):
        """
        Visualiza el entorno en la consola.
        """
        grid = [[' ' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        for (i, j) in self.obstacles:
            grid[i][j] = 'X'  # Obstáculo
        
        pi, pj = self.police_pos
        ti, tj = self.thief_pos
        
        grid[pi][pj] = 'P'  # Policía
        grid[ti][tj] = 'T'  # Ladrón
        
        print("-" * (self.grid_width * 4 +1))
        for row in grid:
            print("| " + " | ".join(row) + " |")
            print("-" * (self.grid_width * 4 +1))

# Implementación del Agente SARSA con Tabla Q Unificada
class SarsaAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, 
                 exploration_rate=1.0, exploration_decay=0.995, min_exploration=0.01):
        self.state_size = state_size  # Número total de estados
        self.action_size = action_size  # Número de acciones
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = min_exploration
        
        # Inicializar la tabla Q: (estado * 2) x acción
        # 2 representa los dos tipos de agentes: 0=Policía, 1=Ladrón
        self.q_table = np.zeros((state_size * 2, action_size))
    
    def choose_action(self, state, agent_type):
        """
        Selecciona una acción usando la estrategia ε-greedy.
        """
        state_agent = state * 2 + agent_type  # Estado único para tipo de agente
        if random.uniform(0,1) < self.epsilon:
            return random.randint(0, self.action_size -1)
        else:
            return np.argmax(self.q_table[state_agent])
    
    def learn(self, state, agent_type, action, reward, next_state, next_action, done):
        """
        Actualiza la tabla Q usando la regla de actualización de SARSA.
        """
        state_agent = state * 2 + agent_type
        next_state_agent = next_state * 2 + agent_type
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.q_table[next_state_agent, next_action]
        
        # Actualización Q
        self.q_table[state_agent, action] += self.lr * (target - self.q_table[state_agent, action])
        
        # Decaimiento de la tasa de exploración
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# Funciones Auxiliares para la Conversión de Estados

# Parámetros del entorno
GRID_SIZE = (5,5)  # Puedes cambiar el tamaño de la grilla
OBSTACLES = [(1,1), (2,2), (3,3)]  # Definir obstáculos

# Crear el entorno
env = PoliceThiefGridEnv(grid_size=GRID_SIZE, obstacles=OBSTACLES)

# Definición de state_size considerando solo una posición por agente
# Cada posición puede ser una de grid_height * grid_width
state_size = env.grid_height * env.grid_width

# Número de acciones
action_size = env.action_space.n

# Inicializar el agente
agent = SarsaAgent(state_size=state_size, action_size=action_size)

# Función para convertir solo la posición del agente a un índice único
def state_to_index(agent_pos_idx):
    return agent_pos_idx

# Función para convertir índice a posiciones (x, y)
def index_to_positions(agent_pos_idx):
    return env._index_to_pos(agent_pos_idx)

# Mapeo de acciones a palabras
action_mapping = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

# Entrenamiento del Agente
# Parámetros de entrenamiento
NUM_EPISODES = 1000
MAX_STEPS = 100

for episode in range(NUM_EPISODES):
    state = env.reset()
    # Convertir el estado a índices únicos
    police_idx, thief_idx = state
    police_state_idx = state_to_index(police_idx)
    thief_state_idx = state_to_index(thief_idx)
    
    done = False
    step = 0
    
    # Inicializar acciones
    police_action = agent.choose_action(police_state_idx, agent_type=0)
    thief_action = agent.choose_action(thief_state_idx, agent_type=1)
    
    while not done and step < MAX_STEPS:
        # ---- Acción del Policía ----
        next_state, reward, done, _ = env.step(police_action, agent_type=0)
        police_next_idx, thief_next_idx = next_state
        police_next_state_idx = state_to_index(police_next_idx)
        
        # Elegir la siguiente acción para el policía (SARSA)
        if not done:
            police_next_action = agent.choose_action(police_next_state_idx, agent_type=0)
        else:
            police_next_action = None
        
        # Aprender del movimiento del policía
        agent.learn(police_state_idx, agent_type=0, action=police_action, reward=reward, 
                    next_state=police_next_state_idx, next_action=police_next_action if police_next_action is not None else 0, done=done)
        
        police_state_idx = police_next_state_idx
        police_action = police_next_action if police_next_action is not None else 0
        
        if done:
            break
        
        # ---- Acción del Ladrón ----
        next_state, reward, done, _ = env.step(thief_action, agent_type=1)
        police_next_idx, thief_next_idx = next_state
        thief_next_state_idx = state_to_index(thief_next_idx)
        
        # Elegir la siguiente acción para el ladrón (SARSA)
        if not done:
            thief_next_action = agent.choose_action(thief_next_state_idx, agent_type=1)
        else:
            thief_next_action = None
        
        # Aprender del movimiento del ladrón
        agent.learn(thief_state_idx, agent_type=1, action=thief_action, reward=reward, 
                    next_state=thief_next_state_idx, next_action=thief_next_action if thief_next_action is not None else 0, done=done)
        
        thief_state_idx = thief_next_state_idx
        thief_action = thief_next_action if thief_next_action is not None else 0
        
        step +=1
    
    # Opcional: Imprimir el progreso cada 100 episodios
    if (episode+1) % 100 == 0:
        print(f"Episode {episode+1}/{NUM_EPISODES} completed. Epsilon: {agent.epsilon:.4f}")

# Función para extraer la política de la tabla Q con coordenadas
def extract_policy(q_table, grid_size, obstacles):
    policies = []
    for agent_type in [0, 1]:  # 0=Policía, 1=Ladrón
        # Iterar solo sobre las posiciones relevantes para el tipo de agente
        for agent_pos_idx in range(env.grid_height * env.grid_width):
            # Convertir el índice del agente a posición
            agent_i = agent_pos_idx // env.grid_width
            agent_j = agent_pos_idx % env.grid_width
            agent_pos = (agent_i, agent_j)
            
            # Verificar si la posición del agente es un obstáculo
            if agent_pos in obstacles:
                best_action = 'X'  # Obstáculo
            else:
                # Obtener el índice en la Q-table
                state_agent = state_to_index(agent_pos_idx) * 2 + agent_type
                best_action_idx = np.argmax(q_table[state_agent])
                best_action = action_mapping.get(best_action_idx, ' ')
            
            policies.append({
                'State': list(agent_pos),       # [x, y] de la posición del agente
                'Agent_Type': agent_type,       # 0=Policía, 1=Ladrón
                'Best_Action': best_action
            })
    return pd.DataFrame(policies)

# Función para crear un grid de la política para un tipo de agente
def create_policy_grid(policy_df, agent_type, grid_size, obstacles):
    grid = [[' ' for _ in range(grid_size[1])] for _ in range(grid_size[0])]
    for index, row in policy_df.iterrows():
        if row['Agent_Type'] != agent_type:
            continue
        state = row['State']
        action = row['Best_Action']
        # state es una lista [x, y]
        pos_i, pos_j = state
        if (pos_i, pos_j) in obstacles:
            grid[pos_i][pos_j] = 'X'
        else:
            grid[pos_i][pos_j] = action
    return grid

# Función para imprimir el grid de la política
def print_policy_grid(grid, title):
    print(title)
    print("-" * (len(grid[0]) * 12 +1))
    for row in grid:
        # Añadir espacios para mejorar la legibilidad
        print("| " + " | ".join(row) + " |")
        print("-" * (len(grid[0]) * 12 +1))

# Extraer la política combinada con coordenadas
policy_df = extract_policy(agent.q_table, GRID_SIZE, OBSTACLES)

# Mostrar la política en forma de tabla
print("Tabla de Política Combinada (Policía y Ladrón):")
display(policy_df.head(20))  # Mostrar las primeras 20 filas para brevedad

# Crear grids para policía y ladrón
police_policy_grid = create_policy_grid(policy_df, agent_type=0, grid_size=GRID_SIZE, obstacles=OBSTACLES)
thief_policy_grid = create_policy_grid(policy_df, agent_type=1, grid_size=GRID_SIZE, obstacles=OBSTACLES)

# Mostrar políticas
print_policy_grid(police_policy_grid, "Policía - Mejor Acción por Posición:")
print_policy_grid(thief_policy_grid, "Ladrón - Mejor Acción por Posición:")

# Exportar la Política a JSON
# Convertir el DataFrame de políticas a una lista de diccionarios
policy_json = policy_df.to_dict(orient='records')

# Guardar el JSON en un archivo llamado 'policy.json' sin escapar caracteres ASCII
with open('policy.json', 'w', encoding='utf-8') as f:
    json.dump(policy_json, f, indent=4, ensure_ascii=False)

print("Política exportada exitosamente a 'policy.json'.")

# Descargar el archivo JSON
files.download('policy.json')
