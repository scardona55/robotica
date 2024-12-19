from flask import Flask, jsonify, request, Response
import cv2
import numpy as np
import math
import threading
import queue
import time
import logging
import json  # Para manejar la política
import random
from collections import deque
import gym
from gym import spaces
import pandas as pd
import os
import comunicacionBluetooth

# Asegúrate de tener instalados los siguientes paquetes:
# pip install flask opencv-python numpy gym pandas

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicializar variables globales para la cámara y la última imagen capturada
URL = "http://192.168.37.118:4747/video"  # URL de la cámara IP (ejemplo: DroidCam)
camera = cv2.VideoCapture(URL)  # Abrir la cámara
# Si prefieres usar una cámara local, descomenta la siguiente línea y comenta la anterior
# camera = cv2.VideoCapture(0)

if not camera.isOpened():
    logger.error("No se pudo abrir la cámara. Verifica la URL o el índice de la cámara.")
    exit(1)

latest_frame = None
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Parámetros de la cuadrícula
rows = 5  # Puedes ajustar este valor según tu configuración
cols = 5  # Puedes ajustar este valor según tu configuración

# Definición del Laberinto (Maze) sin obstáculos por defecto
maze = [
    [0 for _ in range(cols)] for _ in range(rows)
]

# Roles de los marcadores ArUco
# 8 y 9 son IDs de marcadores específicos
roles = {
    8: 0,  # El valor 8 será el Policía inicialmente
    9: 1   # El valor 9 será el Ladrón inicialmente
}
bandera = 0  # Variable de control para alternar roles

# Variable para controlar el estado de ejecución del robot
robot_running = True

app = Flask(__name__)

# ==============================
# Clase para Capturar Frames
# ==============================

class FrameGrabber(threading.Thread):
    def __init__(self, cap, frame_queue):
        super().__init__()
        self.cap = cap
        self.frame_queue = frame_queue
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Error al capturar el video en FrameGrabber.")
                self.running = False
                break
            if not self.frame_queue.full():
                self.frame_queue.put(frame)

    def stop(self):
        self.running = False

# ==============================
# Entorno Personalizado para Gym
# ==============================

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

# ==============================
# Clase Q-Learning Agent
# ==============================

class QLearningAgent:
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
    
    def learn(self, state, agent_type, action, reward, next_state, done):
        """
        Actualiza la tabla Q usando la regla de actualización de Q-Learning.
        """
        state_agent = state * 2 + agent_type
        if done:
            target = reward
        else:
            next_state_agent = next_state * 2 + agent_type
            target = reward + self.gamma * np.max(self.q_table[next_state_agent])
        
        # Actualización Q
        self.q_table[state_agent, action] += self.lr * (target - self.q_table[state_agent, action])
        
        # Decaimiento de la tasa de exploración
        if done:
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# ==============================
# Clase para Controlar el Estado del Robot
# ==============================

class RobotController:
    def __init__(self, maze, qr_detector, policy_dict):
        self.maze = maze
        self.qr_detector = qr_detector
        self.policy_dict = policy_dict  # Política cargada
        self.rows = len(maze)
        self.cols = len(maze[0])
        
        # Información de posición y navegación
        self.current_row = None
        self.current_col = None
        self.current_angle = None
        self.target_row = 0 
        self.target_col = 0
        
        # Estado de navegación
        self.path = []  
        self.intentos_alineacion = 0
        self.movimiento_en_curso = False

        # Modo de operación
        self.use_policy = False  # Inicialmente no usa política

        # Hilo para ejecutar la política
        self.policy_thread = threading.Thread(target=self.run_policy, daemon=True)
        self.policy_thread.start()

    def calcular_camino_optimo(self):
        if self.current_row is None or self.current_col is None:
            logging.error("Posición actual no definida")
            return []

        inicio = (self.current_row, self.current_col)
        objetivo = (self.target_row, self.target_col)

        queue_bfs = deque([(inicio, [])])
        visited = set([inicio])

        while queue_bfs:
            (row, col), path = queue_bfs.popleft()

            if (row, col) == objetivo:
                self.path = path + [(row, col)]
                return self.path

            # Definir los movimientos posibles (arriba, abajo, izquierda, derecha)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc

                # Comprobar si la nueva posición está dentro de los límites
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    if (new_row, new_col) not in visited and self.maze[new_row][new_col] == 0:
                        visited.add((new_row, new_col))
                        queue_bfs.append(((new_row, new_col), path + [(row, col)]))

        # Si no se encontró un camino, devolver una lista vacía
        logging.error("No se encontró un camino al objetivo")
        return []

    def ajustar_angulo(self, objetivo_angulo):
        margen_tolerancia = 20
        diff = (objetivo_angulo - self.current_angle + 360) % 360

        # Verificar si el ángulo ya está alineado
        if diff <= margen_tolerancia or diff >= 360 - margen_tolerancia:
            return self.current_angle

        # Determinar la dirección óptima del giro
        if diff > 180:
            turn_left()  # Giro más corto hacia la izquierda
            self.current_angle = (self.current_angle - 90) % 360
        else:
            turn_right()  # Giro más corto hacia la derecha
            self.current_angle = (self.current_angle + 90) % 360

        return self.current_angle

    def verificar_posicion(self, image):
        """
        Verifica si el robot ha llegado a la casilla objetivo usando detect_qr_in_image.
        """
        # Detectar QR en la imagen capturada
        detected_qrs = detect_shapes_in_image(image, self.rows, self.cols, self.qr_detector)
        
        # Si no se detecta ningún QR, no actualizar posición
        if not detected_qrs:
            logger.warning("No se detectó ningún QR.")
            return False

        # Actualizar posición y ángulo basados en el primer QR detectado
        qr_data = detected_qrs[0]  # Asumimos que el primer QR es el relevante
        self.update_position_and_angle(qr_data)

        # Comprobar si ha llegado a la casilla de destino
        if self.current_row == self.target_row and self.current_col == self.target_col:
            logger.info(f"El robot ha llegado a la posición objetivo: ({self.target_row}, {self.target_col}).")
            # Cambiar al modo de usar política
            self.use_policy = True
            return True

        return False

    def mover_hacia_objetivo(self, image):
        """
        Mueve el robot hacia el objetivo paso a paso o sigue la política si está en modo policy.
        """
        if self.use_policy:
            # Modo de política (ejecutado en un hilo separado)
            pass
        else:
            # Modo de camino óptimo
            if not self.path:
                self.path = self.calcular_camino_optimo()
                if not self.path:
                    logger.error("No hay camino óptimo para moverse.")
                    return

            while not self.verificar_posicion(image) and self.path:
                direccion = self.decidir_movimiento()
                if direccion == "OBJETIVO_ALCANZADO":
                    logging.info("Objetivo alcanzado.")
                    break

                logging.info(f"Moviendo hacia: {direccion}")
                estado = self.ejecutar_movimiento(direccion)

                if estado == "CONTINUAR":
                    continue
                elif estado == "TERMINADO":
                    logging.info("Movimiento completado.")
                    break

                # Pausa para simular movimiento
                time.sleep(0.5)

    def decidir_movimiento(self):
        """
        Decide el siguiente movimiento basado en el camino calculado
        """
        # Si no hay camino, calcularlo
        if not self.path:
            self.path = self.calcular_camino_optimo()
            
            # Si no se encontró camino o ya estamos en el origen
            if not self.path:
                return "OBJETIVO_ALCANZADO"

        # Obtener el siguiente paso
        next_pos = self.path[0]
        dx = next_pos[0] - self.current_row
        dy = next_pos[1] - self.current_col

        # Determinar dirección de movimiento
        if dx == -1:
            return "ARRIBA"
        elif dx == 1:
            return "ABAJO"
        elif dy == -1:
            return "IZQUIERDA"
        elif dy == 1:
            return "DERECHA"

        return "OBJETIVO_ALCANZADO"

    def ejecutar_movimiento(self, direccion):
        """
        Ejecuta el movimiento y actualiza el camino
        """
        logger.info(f"Ejecutando movimiento hacia: {direccion}")
        if direccion == "ARRIBA":
            self.ajustar_angulo(90)
            move_forward()
        elif direccion == "ABAJO":
            self.ajustar_angulo(270)
            move_forward()
        elif direccion == "IZQUIERDA":
            self.ajustar_angulo(180)
            turn_left()
        elif direccion == "DERECHA":
            self.ajustar_angulo(0)
            turn_right()
        
        if self.path:
            self.path.pop(0)

        return "CONTINUAR" if self.path else "TERMINADO"

    def mover_según_politica(self):
        """
        Mueve el robot según la política cargada.
        """
        while self.use_policy and robot_running:
            current_state = self._get_state_index()
            if current_state is None:
                logger.error("Estado actual no definido para la política.")
                break

            # Obtener el tipo de agente actual
            agent_type = current_agent_type  # Variable global

            # Obtener la mejor acción de la política
            key = f"{agent_type}_{current_state}"
            best_action = self.policy_dict.get(key, "OBJETIVO_ALCANZADO")

            if best_action == "up":
                direccion = "ARRIBA"
            elif best_action == "down":
                direccion = "ABAJO"
            elif best_action == "left":
                direccion = "IZQUIERDA"
            elif best_action == "right":
                direccion = "DERECHA"
            else:
                direccion = "OBJETIVO_ALCANZADO"

            if direccion != "OBJETIVO_ALCANZADO":
                logger.info(f"Política: Moviendo hacia {direccion}")
                self.ejecutar_movimiento(direccion)
            else:
                logger.info("Política: No se requiere movimiento.")
                # Aquí podrías implementar alguna acción específica cuando no se requiere movimiento
                break

            # Pausa para simular tiempo de movimiento
            time.sleep(0.5)

    def run_policy(self):
        """
        Ejecuta la política en un hilo separado.
        """
        while robot_running:
            if self.use_policy:
                self.mover_según_politica()
            time.sleep(1)  # Intervalo entre verificaciones

    def _get_state_index(self):
        """
        Obtiene el índice del estado actual para la política.
        """
        if self.current_row is None or self.current_col is None:
            return None
        return self.current_row * self.cols + self.current_col

    def update_position_and_angle(self, qr_data):
        """
        Actualiza la posición y ángulo del robot
        """
        self.current_row = qr_data['row']
        self.current_col = qr_data['col']
        self.current_angle = qr_data['angle']
        
        logging.info(f"Posición actual: fila {self.current_row}, columna {self.current_col}")
        logging.info(f"Ángulo actual: {self.current_angle:.2f}°")

# ==============================
# Funciones de Generación y Dibujo del Laberinto
# ==============================

def maze_generate(filas, columnas, obstacles=[]):
    laberinto = [[0 for _ in range(columnas)] for _ in range(filas)]
    # Si deseas agregar obstáculos de manera dinámica, descomenta lo siguiente
    # for (i, j) in obstacles:
    #     if 0 <= i < filas and 0 <= j < columnas:
    #         laberinto[i][j] = 1
    return laberinto

def draw_grid(image, rows, cols, thickness=1):
    height, width, _ = image.shape
    cell_height = height // rows
    cell_width = width // cols

    for i in range(1, rows):
        cv2.line(image, (0, i * cell_height), (width, i * cell_height), (0, 255, 0), thickness)
    for j in range(1, cols):
        cv2.line(image, (j * cell_width, 0), (j * cell_width, height), (0, 255, 0), thickness)
    return image

def calculate_angle(corners):
    top_left = corners[0]
    top_right = corners[1]
    delta_y = top_right[1] - top_left[1]
    delta_x = top_right[0] - top_left[0]
    angle = np.arctan2(delta_y, delta_x)
    return np.degrees(angle)

def normalize_angle(angle):
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle

# ==============================
# Funciones de Movimiento del Robot
# ==============================

def move_forward():
    for i in range(0,10):
        comunicacionBluetooth.send_command('w')
    logger.info("Movimiento hacia adelante ejecutado.")

def move_back():
    for i in range(0,10):
        comunicacionBluetooth.send_command('s')
    logger.info("Movimiento hacia atrás ejecutado.")

def turn_left():
    for i in range(0, 40):
        comunicacionBluetooth.send_command('d')  # Envía el comando
        # time.sleep(0.1)  # Descomenta si necesitas pausas
    logger.info("Giro a la izquierda ejecutado.")

def turn_right():
    for i in range(0, 40):
        comunicacionBluetooth.send_command('a')  # Envía el comando
        # time.sleep(0.1)  # Descomenta si necesitas pausas
    logger.info("Giro a la derecha ejecutado.")

# ==============================
# Funciones de Dibujo Adicionales
# ==============================

def draw_arrows(image, center, angle):
    # Punto inicial de las flechas
    start_point = tuple(map(int, center))

    # Punto final para la flecha roja (0° referencia)
    end_point_red = (int(center[0] + 50), int(center[1]))

    # Punto final para la flecha azul (dirección del marcador)
    angle_rad = math.radians(angle)
    end_point_blue = (int(center[0] + 50 * math.cos(angle_rad)),
                      int(center[1] - 50 * math.sin(angle_rad)))

    # Dibujar flechas
    cv2.arrowedLine(image, start_point, end_point_red, (0, 0, 255), 2, tipLength=0.3)  # Roja (0°)
    cv2.arrowedLine(image, start_point, end_point_blue, (0, 255, 0), 2, tipLength=0.3)  # Azul (ángulo)

# ==============================
# Funciones de Detección y Validación
# ==============================

def detect_shapes_in_image(image, rows, cols, qr_detector):
    detected_shapes = []

    # Calcular la fila y columna de la cuadrícula
    height, width = image.shape[:2]
    cell_width = width / cols
    cell_height = height / rows

    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar marcadores ArUco
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Dibujar marcadores detectados
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            # Calcular el centro del marcador
            center = np.mean(marker_corners[0], axis=0)

            # Calcular la inclinación del marcador
            angle = calculate_angle(marker_corners[0])
            angle = normalize_angle(angle)

            # Dibujar flechas indicando orientación
            draw_arrows(image, center, angle)

            # Mostrar ID y ángulo en la imagen
            cv2.putText(image, f"ID: {marker_id}", (int(center[0] - 50), int(center[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"{int(angle)}°", (int(center[0] - 50), int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Determinar la posición en la cuadrícula
            row = int(math.floor(center[1] // cell_height))
            col = int(math.floor(center[0] // cell_width))
            cell_index = row * cols + col
            cell_center_x = int((col + 0.5) * cell_width)
            cell_center_y = int((row + 0.5) * cell_height)

            # Mostrar el índice de la celda
            cv2.putText(image, f"{cell_index}", (int(center[0]-10), int(center[1]+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Asignar roles basados en el ID y la bandera
            role = -1
            if marker_id in roles:
                role = roles[marker_id]
            roleTxt = "Policia" if role == 0 else "Ladron" if role ==1 else "Indefinido"
            cv2.putText(image, f"{roleTxt}", (int(center[0] - 10), int(center[1] + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Añadir información del marcador detectado a la lista
            detected_shapes.append({
                "shape": marker_id,
                "angle": angle,
                "x": math.floor(center[0]),
                "y": math.floor(center[1]),
                "cell_center_x": cell_center_x,
                "cell_center_y": cell_center_y,
                "cell_index": cell_index,
                "row": row,
                "col": col,
                "cell_width": cell_width,
                "cell_height": cell_height,
                "role": role
            })

    logger.debug(f"Marcadores detectados: {detected_shapes}")
    return detected_shapes

def validate_and_convert_dict(input_data):
    """
    Recorre los valores de un diccionario o lista de diccionarios y valida que sean serializables a JSON.
    Convierte los tipos numpy a tipos nativos de Python cuando sea necesario.
    """
    if isinstance(input_data, dict):  # Si es un diccionario, recorremos sus items
        for key, value in input_data.items():
            if isinstance(value, np.generic):  # Si el valor es un tipo numpy no serializable
                input_data[key] = value.item()  # Convertir a un tipo Python nativo
            elif isinstance(value, (dict, list)):  # Si es un diccionario o lista, validamos recursivamente
                input_data[key] = validate_and_convert_dict(value)
    elif isinstance(input_data, list):  # Si es una lista, recorremos sus elementos
        for i, item in enumerate(input_data):
            if isinstance(item, np.generic):  # Si el valor es un tipo numpy no serializable
                input_data[i] = item.item()  # Convertir a un tipo Python nativo
            elif isinstance(item, (dict, list)):  # Si el elemento es un diccionario o lista, validamos recursivamente
                input_data[i] = validate_and_convert_dict(item)

    return input_data

# ==============================
# Función para Generar la Política usando Q-Learning
# ==============================

def generate_policy(grid_size, obstacles=[]):
    """
    Entrena un agente de Q-Learning en el entorno personalizado y genera la política.
    Retorna una lista de diccionarios que representan la política.
    """
    # Crear el entorno
    env = PoliceThiefGridEnv(grid_size=grid_size, obstacles=obstacles)

    # Definición de state_size considerando solo una posición por agente
    # Cada posición puede ser una de grid_height * grid_width
    state_size = env.grid_height * env.grid_width

    # Número de acciones
    action_size = env.action_space.n

    # Inicializar el agente
    agent = QLearningAgent(state_size=state_size, action_size=action_size)

    # Función para convertir solo la posición del agente a un índice único
    def state_to_index(agent_pos_idx):
        return agent_pos_idx

    # Función para convertir índice a posiciones (x, y)
    def index_to_positions(agent_pos_idx):
        return env._index_to_pos(agent_pos_idx)

    # Entrenamiento del Agente
    # Parámetros de entrenamiento
    NUM_EPISODES = 1000
    MAX_STEPS = 100

    # Mapeo de acciones a palabras
    action_mapping = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    logger.info("Iniciando entrenamiento de Q-Learning...")
    for episode in range(NUM_EPISODES):
        state = env.reset()
        # Convertir el estado a índices únicos
        police_idx, thief_idx = state
        police_state_idx = state_to_index(police_idx)
        thief_state_idx = state_to_index(thief_idx)
        
        done = False
        step = 0
        
        while not done and step < MAX_STEPS:
            # ---- Acción del Policía ----
            police_action = agent.choose_action(police_state_idx, agent_type=0)
            next_state, reward, done, _ = env.step(police_action, agent_type=0)
            police_next_idx, thief_next_idx = next_state
            police_next_state_idx = state_to_index(police_next_idx)
            
            # Aprender del movimiento del policía
            agent.learn(police_state_idx, agent_type=0, action=police_action, reward=reward, 
                        next_state=police_next_state_idx, done=done)
            
            police_state_idx = police_next_state_idx
            
            if done:
                break
            
            # ---- Acción del Ladrón ----
            thief_action = agent.choose_action(thief_state_idx, agent_type=1)
            next_state, reward, done, _ = env.step(thief_action, agent_type=1)
            police_next_idx, thief_next_idx = next_state
            thief_next_state_idx = state_to_index(thief_next_idx)
            
            # Aprender del movimiento del ladrón
            agent.learn(thief_state_idx, agent_type=1, action=thief_action, reward=reward, 
                        next_state=thief_next_state_idx, done=done)
            
            thief_state_idx = thief_next_state_idx
            step +=1
        
        # Opcional: Imprimir el progreso cada 100 episodios
        if (episode+1) % 100 == 0:
            logger.info(f"Episode {episode+1}/{NUM_EPISODES} completado. Epsilon: {agent.epsilon:.4f}")

    logger.info("Entrenamiento completado.")

    # Función para extraer la política de la tabla Q con coordenadas
    def extract_policy(q_table, grid_size, obstacles, action_mapping):
        policies = []
        for agent_type in [0, 1]:  # 0=Policía, 1=Ladrón
            # Iterar sobre todas las posiciones
            for agent_pos_idx in range(grid_size[0] * grid_size[1]):
                # Convertir el índice del agente a posición
                agent_i = agent_pos_idx // grid_size[1]
                agent_j = agent_pos_idx % grid_size[1]
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
                    'State': [agent_i, agent_j],       # [x, y] de la posición del agente
                    'Agent_Type': agent_type,          # 0=Policía, 1=Ladrón
                    'Best_Action': best_action
                })
        return policies

    # Extraer la política combinada con coordenadas
    policy_list = extract_policy(agent.q_table, grid_size=env.grid_height, obstacles=env.obstacles, action_mapping=action_mapping)

    logger.info("Política generada exitosamente.")

    return policy_list

# ==============================
# Funciones Auxiliares Adicionales
# ==============================

def fill_cells(frame, matrix, alpha=0.7):
    """Rellena de color rojo translúcido las celdas correspondientes a los obstáculos."""
    rows, cols = len(matrix), len(matrix[0])
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    overlay = frame.copy()

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                x1, y1 = j * cell_width, i * cell_height
                x2, y2 = x1 + cell_width, y1 + cell_height
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

def highlight_start_end(frame, rows, cols):
    """Resalta en verde la celda de inicio y en azul la celda de fin."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    overlay = frame.copy()
    alpha = 0.5

    # Inicio (0,0) - Verde
    cv2.rectangle(overlay, (0, 0), (cell_width, cell_height), (0, 255, 0), -1)

    # Final (rows-1, cols-1) - Azul
    cv2.rectangle(overlay, ((cols - 1) * cell_width, (rows - 1) * cell_height),
                  ((cols) * cell_width, (rows) * cell_height), (255, 0, 0), -1)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

# ==============================
# Rutas de la API
# ==============================

@app.route('/cambiar_roles', methods=['POST'])
def cambiar_roles():
    global bandera, roles, current_agent_type, robot

    # Cambiar el valor de la bandera (0 a 1 o 1 a 0)
    bandera = 1 - bandera

    # Cambiar los roles en función del valor de la bandera
    if bandera == 0:
        roles = {8: 0, 9: 1}  # Policía es 0 y Ladrón es 1
        current_agent_type = 0
    else:
        roles = {8: 1, 9: 0}  # Policía es 1 y Ladrón es 0
        current_agent_type = 1

    logger.info(f"Roles actualizados: {roles}")
    logger.info(f"Rol actual del agente: {'Policía' if current_agent_type ==0 else 'Ladrón'}")

    return jsonify({
        'bandera': bandera,
        'roles': roles,
        'current_agent_type': current_agent_type
    })

@app.route('/video_feed', methods=['GET'])
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                # Copiar el frame para no modificar la referencia global
                frame_copy = latest_frame.copy()
                # Dibujar la grilla y detectar formas
                detected_shapes = detect_shapes_in_image(frame_copy, rows, cols, qr_detector)
                draw_grid(frame_copy, rows, cols)
                fill_cells(frame_copy, maze)
                highlight_start_end(frame_copy, rows, cols)
                # Codificar la imagen como JPEG
                _, buffer = cv2.imencode('.jpg', frame_copy)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                time.sleep(0.01)  # Esperar un poco si no hay frame disponible

    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/maze', methods=['GET'])
def get_maze():
    return jsonify(maze)

@app.route('/detect_shapes', methods=['GET'])
def detect_shapes_endpoint():
    global latest_frame
    if latest_frame is not None:
        # Copiar el frame para no modificar la referencia global
        frame_copy = latest_frame.copy()
        # Dibujar la grilla y detectar formas
        shapes = detect_shapes_in_image(frame_copy, rows, cols, qr_detector)
        shapes = validate_and_convert_dict(shapes)
        return jsonify(shapes)
    else:
        return jsonify({"error": "No frame available from the camera"}), 500

# ==============================
# Función Principal de Servidor Flask
# ==============================

def main():
    global latest_frame, robot_running, roles, current_agent_type

    # Generar y cargar la política
    policy_path = 'policy.json'
    if os.path.exists(policy_path):
        # Cargar la política desde 'policy.json'
        try:
            with open(policy_path, 'r', encoding='utf-8') as f:
                policy_data = json.load(f)
            logger.info("Política cargada exitosamente desde 'policy.json'.")
        except json.JSONDecodeError:
            logger.error("Error al decodificar 'policy.json'. Generando una nueva política.")
            policy_data = generate_policy(grid_size=(rows, cols), obstacles=[])
            # Guardar la política generada
            with open(policy_path, 'w', encoding='utf-8') as f:
                json.dump(policy_data, f, indent=4, ensure_ascii=False)
            logger.info("Política generada y guardada en 'policy.json'.")
    else:
        logger.info("'policy.json' no encontrado. Iniciando entrenamiento de Q-Learning...")
        # Generar la política
        policy_data = generate_policy(grid_size=(rows, cols), obstacles=[])
        # Guardar la política generada
        with open(policy_path, 'w', encoding='utf-8') as f:
            json.dump(policy_data, f, indent=4, ensure_ascii=False)
        logger.info("Política generada y guardada en 'policy.json'.")

    # Convertir la política a un diccionario para acceso rápido
    policy_dict = {}
    for entry in policy_data:
        agent_type = entry.get('Agent_Type')
        state = entry.get('State')
        best_action = entry.get('Best_Action')

        if agent_type is None or state is None or best_action is None:
            continue  # Ignorar entradas incompletas

        state_index = state[0] * cols + state[1]
        key = f"{agent_type}_{state_index}"  # Clave única por tipo de agente y estado
        policy_dict[key] = best_action

    # Inicializar el detector QR
    qr_detector = cv2.QRCodeDetector()

    # Inicializar el RobotController con el laberinto y la política
    robot = RobotController(maze, qr_detector, policy_dict)

    # Inicializar el tipo de agente actual (0=Policía, 1=Ladrón)
    current_agent_type = 0  # Inicialmente Policía

    # Iniciar el FrameGrabber
    frame_queue = queue.Queue(maxsize=50)
    frame_grabber = FrameGrabber(camera, frame_queue)
    frame_grabber.start()

    # Iniciar el servidor Flask en un hilo separado para evitar bloqueos
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                latest_frame = frame  # Actualizar el frame más reciente

                # Procesar el frame para detección y visualización
                detected_shapes = detect_shapes_in_image(frame, rows, cols, qr_detector)
                draw_grid(frame, rows, cols)
                fill_cells(frame, maze)
                highlight_start_end(frame, rows, cols)

                # Mostrar el frame en una ventana
                cv2.imshow('Cuadrícula con análisis', frame)

                # Actualizar la posición del robot si hay marcadores detectados
                if detected_shapes:
                    robot.verificar_posicion(frame)

                # Decidir y ejecutar el siguiente movimiento
                robot.mover_hacia_objetivo(frame)

                # Controlar la salida con la tecla 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Presionaste 'q'. Cerrando conexión y terminando programa...")
                    break
            else:
                time.sleep(0.01)  # Evitar uso intensivo de CPU

    except KeyboardInterrupt:
        logger.info("Interrupción por teclado. Cerrando programa...")

    finally:
        # Detener el FrameGrabber y el hilo del robot
        robot_running = False
        robot.policy_thread.join()
        frame_grabber.stop()
        frame_grabber.join()

        # Liberar recursos
        camera.release()
        cv2.destroyAllWindows()

# ==============================
# Ejecutar el Servidor
# ==============================

if __name__ == "__main__":
    main()
