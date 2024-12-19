from flask import Flask, jsonify, request, Response
import cv2
import numpy as np
import math
import threading
import logging
import requests
import pickle
import os
import time

# Importar el módulo de comunicación Bluetooth
import comunicacionBluetooth

app = Flask(__name__)

# Configuración del Logging
logging.basicConfig(
    level=logging.INFO,  # Cambia a DEBUG para más detalles
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicializar variables globales para la cámara y la última imagen capturada
camera = cv2.VideoCapture("http://10.144.145.9:4747/video")  # Abrir la cámara
# camera = cv2.VideoCapture(0)  # Alternativa si usas cámara local
latest_frame = None
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
rows = 5
cols = 5
maze = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
]
roles = {
    8: 0,  # El valor 8 será el policía
    9: 1   # El valor 9 será el ladrón
}
bandera = 0

# ==============================
# Definición de la Clase QLearningAgent
# ==============================

class QLearningAgent:
    def __init__(self, rows, cols, role, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.1, decay_rate=0.995):
        """
        Inicializa el agente de Q-learning.

        :param rows: Número de filas del mapa.
        :param cols: Número de columnas del mapa.
        :param role: Rol del agente (0 = Policía, 1 = Ladrón).
        :param alpha: Tasa de aprendizaje.
        :param gamma: Factor de descuento.
        :param epsilon: Tasa de exploración inicial.
        :param min_epsilon: Tasa mínima de exploración.
        :param decay_rate: Tasa de decaimiento de epsilon.
        """
        self.rows = rows
        self.cols = cols
        self.role = role
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.actions = [0, 1, 2, 3]  # 0=Arriba, 1=Abajo, 2=Izquierda, 3=Derecha
        self.Q = np.zeros((rows * cols, len(self.actions)))

    def state_to_index(self, row, col):
        return row * self.cols + col

    def choose_action(self, state):
        """
        Elige una acción basada en la política epsilon-greedy.
        """
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(self.actions)
        else:
            action = np.argmax(self.Q[state])
        return action

    def update_Q(self, state, action, reward, next_state):
        """
        Actualiza el valor Q según la ecuación de Q-learning.
        """
        self.Q[state][action] += self.alpha * (reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state][action])

    def get_reward(self, current_position, next_position, maze, target_position, police_position=None):
        """
        Define la función de recompensa basada en el rol del agente.

        :param current_position: Tupla (fila, columna) del estado actual.
        :param next_position: Tupla (fila, columna) del siguiente estado.
        :param maze: Matriz del laberinto.
        :param target_position: Tupla (fila, columna) del objetivo.
        :param police_position: Tupla (fila, columna) de la posición del policía (solo para el ladrón).
        :return: Recompensa numérica.
        """
        row, col = next_position

        # Recompensa por obstáculo
        if maze[row][col] == 1:
            return -10

        # Recompensa por alcanzar el objetivo
        if self.role == 0:  # Policía
            if next_position == target_position:
                return 100
            else:
                return -1  # Penalización por cada paso
        elif self.role == 1:  # Ladrón
            if next_position == target_position:
                return 100
            # Penalización si el policía está cerca
            if police_position:
                distance = np.linalg.norm(np.array(police_position) - np.array(next_position))
                if distance <= 1.5:
                    return -100
            return -1  # Penalización por cada paso

    def train(self, maze, target_position, police_position=None, episodes=1000, max_steps=100):
        """
        Entrena el agente usando Q-learning.

        :param maze: Matriz del laberinto.
        :param target_position: Tupla (fila, columna) del objetivo.
        :param police_position: Tupla (fila, columna) de la posición del policía (solo para el ladrón).
        :param episodes: Número de episodios de entrenamiento.
        :param max_steps: Número máximo de pasos por episodio.
        """
        for episode in range(episodes):
            state_row, state_col = 0, 0  # Estado inicial
            state = self.state_to_index(state_row, state_col)
            total_rewards = 0

            for step in range(max_steps):
                action = self.choose_action(state)

                # Determinar la siguiente posición basada en la acción
                if action == 0 and state_row > 0:  # Arriba
                    next_row, next_col = state_row - 1, state_col
                elif action == 1 and state_row < self.rows - 1:  # Abajo
                    next_row, next_col = state_row + 1, state_col
                elif action == 2 and state_col > 0:  # Izquierda
                    next_row, next_col = state_row, state_col - 1
                elif action == 3 and state_col < self.cols - 1:  # Derecha
                    next_row, next_col = state_row, state_col + 1
                else:
                    next_row, next_col = state_row, state_col  # Acción inválida

                # Calcular la recompensa
                reward = self.get_reward(
                    current_position=(state_row, state_col),
                    next_position=(next_row, next_col),
                    maze=maze,
                    target_position=target_position,
                    police_position=police_position
                )
                total_rewards += reward

                # Calcular el nuevo estado
                next_state = self.state_to_index(next_row, next_col)

                # Actualizar Q-valor
                self.update_Q(state, action, reward, next_state)

                # Actualizar el estado actual
                state = next_state
                state_row, state_col = next_row, next_col

                # Terminar episodio si alcanza el objetivo o es atrapado
                if (self.role == 0 and (state_row, state_col) == target_position) or \
                   (self.role == 1 and (state_row, state_col) == target_position):
                    break

            # Decaimiento de epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay_rate)

            if (episode + 1) % 100 == 0:
                logger.info(f"Episodio {episode + 1}/{episodes} completado. Recompensa total: {total_rewards}")

        logger.info("Entrenamiento completado.")

    def save_policy(self, filename):
        """
        Guarda la matriz Q en un archivo.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self.Q, f)
        logger.info(f"Política guardada en {filename}.")

    def load_policy(self, filename):
        """
        Carga la matriz Q desde un archivo.
        """
        with open(filename, 'rb') as f:
            self.Q = pickle.load(f)
        logger.info(f"Política cargada desde {filename}.")

    def get_action_from_policy(self, state):
        """
        Obtiene la mejor acción basada en la política entrenada.
        """
        return np.argmax(self.Q[state])

# ==============================
# Funciones Auxiliares
# ==============================

def calculate_angle(corners):
    # Extraer las coordenadas de las esquinas del marcador
    top_left, top_right, bottom_right, bottom_left = corners.reshape((4, 2))

    # Calcular el vector desde la esquina superior izquierda (0° referencia)
    vector = top_right - top_left

    # Invertir el eje Y para que el ángulo aumente en sentido horario
    vector[1] = -vector[1]

    # Calcular el ángulo en radianes y convertir a grados
    angle = math.degrees(math.atan2(vector[1], vector[0]))

    # Convertir el ángulo a rango [0, 360]
    if angle < 0:
        angle += 360

    return angle

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

def draw_grid(image, rows, cols, grid_values):
    """
    Dibuja una grilla sobre la imagen dada según el número de filas y columnas.
    Colorea en rojo las casillas correspondientes a un valor de 1 en la matriz grid_values con un alpha de 0.7.

    :param image: Imagen sobre la cual dibujar la grilla.
    :param rows: Número de filas de la grilla.
    :param cols: Número de columnas de la grilla.
    :param grid_values: Matriz de valores (0 y 1) que indican qué celdas colorear.
    """
    height, width = image.shape[:2]
    cell_width = width / cols
    cell_height = height / rows

    # Crear un overlay para el alpha
    overlay = image.copy()

    for row in range(rows):
        for col in range(cols):
            if grid_values[row][col] == 1:
                # Coordenadas de la celda
                x1 = int(col * cell_width)
                y1 = int(row * cell_height)
                x2 = int((col + 1) * cell_width)
                y2 = int((row + 1) * cell_height)

                # Dibujar un rectángulo rojo en la celda correspondiente
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

    # Aplicar el efecto de alpha
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Dibujar líneas horizontales
    for row in range(1, rows):
        y = int(row * cell_height)
        cv2.line(image, (0, y), (width, y), (0, 255, 0), 1)

    # Dibujar líneas verticales
    for col in range(1, cols):
        x = int(col * cell_width)
        cv2.line(image, (x, 0), (x, height), (0, 255, 0), 1)

def detect_shapes_in_image(image, rows, cols):
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

            # Dibujar flechas indicando orientación
            draw_arrows(image, center, angle)

            # Mostrar ID y ángulo en la imagen
            cv2.putText(image, f"ID: {marker_id}", (int(center[0] - 50), int(center[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"{int(angle)}°", (int(center[0] - 50), int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Determinar la posición en la cuadrícula
            row = int(math.floor(center[1]) // cell_height)
            col = int(math.floor(center[0]) // cell_width)
            cell_index = row * cols + col
            cell_center_x = int((col + 0.5) * cell_width)
            cell_center_y = int((row + 0.5) * cell_height)

            cv2.putText(image, f"{cell_index}", (int(center[0]-10), int(center[1]+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Asignar roles según ID y bandera
            role = -1
            if marker_id in roles:
                role = roles[marker_id]
            roleTxt = "Policía" if role == 0 else "Ladrón" if role == 1 else "Desconocido"
            cv2.putText(image, f"{roleTxt}", (int(center[0] - 10), int(center[1] + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

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

    print(detected_shapes)
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

def train_policies_if_needed():
    """
    Entrena las políticas para Policía y Ladrón si no existen archivos de política.
    """
    policia_policy_file = 'policy_policia.pkl'
    ladron_policy_file = 'policy_ladron.pkl'

    # Verificar si las políticas ya están entrenadas
    if not os.path.exists(policia_policy_file) or not os.path.exists(ladron_policy_file):
        logger.info("Iniciando entrenamiento de políticas...")

        # Definir el mapa del laberinto
        maze = [
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        rows = len(maze)
        cols = len(maze[0])

        # Definir objetivos
        policia_target = (4, 4)  # Casilla objetivo para Policía (puedes ajustarlo)
        ladron_target = (4, 4)    # Casilla objetivo para Ladrón (puedes ajustarlo)

        # Inicializar agentes
        agente_policia = QLearningAgent(rows, cols, role=0)
        agente_ladron = QLearningAgent(rows, cols, role=1)

        # Entrenar agente Policía
        logger.info("Entrenando agente Policía...")
        agente_policia.train(
            maze=maze,
            target_position=policia_target,
            episodes=1000
        )
        agente_policia.save_policy(policia_policy_file)

        # Entrenar agente Ladrón
        logger.info("Entrenando agente Ladrón...")
        # Supongamos que el Policía está en (0,0) durante el entrenamiento
        police_position = (0, 0)
        agente_ladron.train(
            maze=maze,
            target_position=ladron_target,
            police_position=police_position,
            episodes=1000
        )
        agente_ladron.save_policy(ladron_policy_file)

        logger.info("Entrenamiento de políticas completado.")
    else:
        logger.info("Políticas ya entrenadas. Cargando políticas existentes.")

def load_policies():
    """
    Carga las políticas de Q-learning para Policía y Ladrón desde archivos.
    """
    policia_policy_file = 'policy_policia.pkl'
    ladron_policy_file = 'policy_ladron.pkl'

    agente_policia = QLearningAgent(rows, cols, role=0)
    agente_ladron = QLearningAgent(rows, cols, role=1)

    agente_policia.load_policy(policia_policy_file)
    agente_ladron.load_policy(ladron_policy_file)

    return agente_policia, agente_ladron

def control_robot(agente_policia, agente_ladron):
    """
    Controla el robot basado en la detección de roles y las políticas de Q-learning.
    """
    # Establecer conexión Bluetooth
    TARGET_MAC = "00:1B:10:21:2C:1B"  # Reemplaza con la dirección MAC de tu dispositivo
    bluetooth_socket = comunicacionBluetooth.bluetooth_connect(TARGET_MAC)
    if not bluetooth_socket:
        logger.error("No se pudo establecer la conexión Bluetooth. Terminando controlador.")
        return

    frame_counter = 0
    COMMAND_INTERVAL = 48  # Envío de comandos cada 48 iteraciones

    while True:
        # Obtener detecciones desde el servidor Flask
        try:
            response = requests.get("http://localhost:5000/detect_shapes")
            detections = response.json()
        except Exception as e:
            logger.error(f"Error al obtener detecciones del servidor: {e}")
            detections = []

        # Determinar el rol del robot basado en sus propios marcadores
        robot_detections = [d for d in detections if d['shape'] in roles]
        if robot_detections:
            robot_role = robot_detections[0]['role']  # 0 = Policía, 1 = Ladrón
            robot_position = (robot_detections[0]['row'], robot_detections[0]['col'])
            logger.info(f"Rol del robot: {'Policía' if robot_role == 0 else 'Ladrón'}, Posición: {robot_position}")
        else:
            logger.warning("No se detectó el rol del robot. Esperando detección...")
            time.sleep(1)
            continue

        # Obtener posiciones de los otros agentes
        if robot_role == 0:
            # Policía: Buscar al Ladrón
            ladrons = [d for d in detections if d['role'] == 1]
            if not ladrons:
                logger.info("Policía no detecta a ningún Ladrón. Esperando...")
                time.sleep(1)
                continue
            # Asumiremos que hay solo un ladrón
            target_position = (ladrons[0]['row'], ladrons[0]['col'])
            state = agente_policia.state_to_index(robot_position[0], robot_position[1])
            action = agente_policia.get_action_from_policy(state)
        elif robot_role == 1:
            # Ladrón: Buscar la casilla más lejana y evitar al Policía
            target_position = (4, 4)  # Adaptar según el mapa real
            policias = [d for d in detections if d['role'] == 0]
            if policias:
                police_position = (policias[0]['row'], policias[0]['col'])
            else:
                police_position = None
            state = agente_ladron.state_to_index(robot_position[0], robot_position[1])
            action = agente_ladron.get_action_from_policy(state)
        else:
            logger.warning(f"Rol desconocido detectado: {robot_role}")
            time.sleep(1)
            continue

        # Mapear la acción a comandos
        accion_map = {
            0: "MOVE_FORWARD",  # Arriba
            1: "MOVE_BACKWARD", # Abajo
            2: "TURN_LEFT",     # Izquierda
            3: "TURN_RIGHT"     # Derecha
        }
        command = accion_map.get(action, None)
        if command:
            comunicacionBluetooth.send_command(command)
            logger.info(f"Comando enviado: {command}")
            frame_counter += 1

        # Verificar si se ha alcanzado el objetivo
        if robot_position == target_position:
            if robot_role == 0:
                logger.info("Policía ha alcanzado al Ladrón. Objetivo cumplido.")
            elif robot_role == 1:
                logger.info("Ladrón ha alcanzado la casilla más lejana. Objetivo cumplido.")
            break

        # Enviar comandos en intervalos
        if frame_counter % COMMAND_INTERVAL == 0:
            if command:
                comunicacionBluetooth.send_command(command)
                logger.info(f"Comando enviado en intervalo: {command}")

        # Esperar antes de la siguiente iteración
        time.sleep(0.5)  # Ajusta según la velocidad del robot

def capture_frames():
    """
    Captura continuamente frames de la cámara y actualiza la variable global latest_frame.
    """
    global latest_frame
    while True:
        ret, frame = camera.read()
        if ret:
            latest_frame = frame

# ==============================
# Funciones para Entrenar y Cargar Políticas
# ==============================

def train_policies_if_needed():
    """
    Entrena las políticas para Policía y Ladrón si no existen archivos de política.
    """
    policia_policy_file = 'policy_policia.pkl'
    ladron_policy_file = 'policy_ladron.pkl'

    # Verificar si las políticas ya están entrenadas
    if not os.path.exists(policia_policy_file) or not os.path.exists(ladron_policy_file):
        logger.info("Iniciando entrenamiento de políticas...")

        # Definir el mapa del laberinto
        maze = [
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        rows = len(maze)
        cols = len(maze[0])

        # Definir objetivos
        policia_target = (4, 4)  # Casilla objetivo para Policía (puedes ajustarlo)
        ladron_target = (4, 4)    # Casilla objetivo para Ladrón (puedes ajustarlo)

        # Inicializar agentes
        agente_policia = QLearningAgent(rows, cols, role=0)
        agente_ladron = QLearningAgent(rows, cols, role=1)

        # Entrenar agente Policía
        logger.info("Entrenando agente Policía...")
        agente_policia.train(
            maze=maze,
            target_position=policia_target,
            episodes=1000
        )
        agente_policia.save_policy(policia_policy_file)

        # Entrenar agente Ladrón
        logger.info("Entrenando agente Ladrón...")
        # Supongamos que el Policía está en (0,0) durante el entrenamiento
        police_position = (0, 0)
        agente_ladron.train(
            maze=maze,
            target_position=ladron_target,
            police_position=police_position,
            episodes=1000
        )
        agente_ladron.save_policy(ladron_policy_file)

        logger.info("Entrenamiento de políticas completado.")
    else:
        logger.info("Políticas ya entrenadas. Cargando políticas existentes.")

def load_policies():
    """
    Carga las políticas de Q-learning para Policía y Ladrón desde archivos.
    """
    policia_policy_file = 'policy_policia.pkl'
    ladron_policy_file = 'policy_ladron.pkl'

    agente_policia = QLearningAgent(rows, cols, role=0)
    agente_ladron = QLearningAgent(rows, cols, role=1)

    agente_policia.load_policy(policia_policy_file)
    agente_ladron.load_policy(ladron_policy_file)

    return agente_policia, agente_ladron

# ==============================
# Endpoints de Flask
# ==============================

@app.route('/cambiar_roles', methods=['POST'])
def cambiar_roles():
    global bandera, roles

    # Cambiar el valor de la bandera (0 a 1 o 1 a 0)
    bandera = 1 - bandera

    # Cambiar los roles en función del valor de la bandera
    if bandera == 0:
        roles = {8: 0, 9: 1}  # Policía es 0 y Ladrón es 1
    else:
        roles = {8: 1, 9: 0}  # Policía es 1 y Ladrón es 0

    return jsonify({
        'bandera': bandera,
        'roles': roles
    })

@app.route('/video_feed', methods=['GET'])
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                # Copiar el frame para no modificar la referencia global
                frame_copy = latest_frame.copy()
                # Dibujar la grilla y detectar formas
                detect_shapes_in_image(frame_copy, rows, cols)
                draw_grid(frame_copy, rows, cols, maze)
                # Codificar la imagen como JPEG
                _, buffer = cv2.imencode('.jpg', frame_copy)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_shapes', methods=['GET'])
def detect_shapes_endpoint():
    global latest_frame
    if latest_frame is not None:
        # Copiar el frame para no modificar la referencia global
        frame_copy = latest_frame.copy()
        # Dibujar la grilla y detectar formas
        shapes = detect_shapes_in_image(frame_copy, rows, cols)
        shapes = validate_and_convert_dict(shapes)
        return jsonify(shapes)
    else:
        return jsonify({"error": "No frame available from the camera"}), 500

@app.route('/maze', methods=['GET'])
def get_maze():
    return jsonify(maze)

# ==============================
# Funciones para Controlar el Robot
# ==============================

def control_robot(agente_policia, agente_ladron):
    """
    Controla el robot basado en la detección de roles y las políticas de Q-learning.
    """
    # Establecer conexión Bluetooth
    TARGET_MAC = "00:1B:10:21:2C:1B"  # Reemplaza con la dirección MAC de tu dispositivo
    bluetooth_socket = comunicacionBluetooth.bluetooth_connect(TARGET_MAC)
    if not bluetooth_socket:
        logger.error("No se pudo establecer la conexión Bluetooth. Terminando controlador.")
        return

    frame_counter = 0
    COMMAND_INTERVAL = 48  # Envío de comandos cada 48 iteraciones

    while True:
        # Obtener detecciones desde el servidor Flask
        try:
            response = requests.get("http://localhost:5000/detect_shapes")
            detections = response.json()
        except Exception as e:
            logger.error(f"Error al obtener detecciones del servidor: {e}")
            detections = []

        # Determinar el rol del robot basado en sus propios marcadores
        robot_detections = [d for d in detections if d['shape'] in roles]
        if robot_detections:
            robot_role = robot_detections[0]['role']  # 0 = Policía, 1 = Ladrón
            robot_position = (robot_detections[0]['row'], robot_detections[0]['col'])
            logger.info(f"Rol del robot: {'Policía' if robot_role == 0 else 'Ladrón'}, Posición: {robot_position}")
        else:
            logger.warning("No se detectó el rol del robot. Esperando detección...")
            time.sleep(1)
            continue

        # Obtener posiciones de los otros agentes
        if robot_role == 0:
            # Policía: Buscar al Ladrón
            ladrons = [d for d in detections if d['role'] == 1]
            if not ladrons:
                logger.info("Policía no detecta a ningún Ladrón. Esperando...")
                time.sleep(1)
                continue
            # Asumiremos que hay solo un ladrón
            target_position = (ladrons[0]['row'], ladrons[0]['col'])
            state = agente_policia.state_to_index(robot_position[0], robot_position[1])
            action = agente_policia.get_action_from_policy(state)
        elif robot_role == 1:
            # Ladrón: Buscar la casilla más lejana y evitar al Policía
            target_position = (4, 4)  # Adaptar según el mapa real
            policias = [d for d in detections if d['role'] == 0]
            if policias:
                police_position = (policias[0]['row'], policias[0]['col'])
            else:
                police_position = None
            state = agente_ladron.state_to_index(robot_position[0], robot_position[1])
            action = agente_ladron.get_action_from_policy(state)
        else:
            logger.warning(f"Rol desconocido detectado: {robot_role}")
            time.sleep(1)
            continue

        # Mapear la acción a comandos
        accion_map = {
            0: "MOVE_FORWARD",  # Arriba
            1: "MOVE_BACKWARD", # Abajo
            2: "TURN_LEFT",     # Izquierda
            3: "TURN_RIGHT"     # Derecha
        }
        command = accion_map.get(action, None)
        if command:
            comunicacionBluetooth.send_command(command)
            logger.info(f"Comando enviado: {command}")
            frame_counter += 1

        # Verificar si se ha alcanzado el objetivo
        if robot_position == target_position:
            if robot_role == 0:
                logger.info("Policía ha alcanzado al Ladrón. Objetivo cumplido.")
            elif robot_role == 1:
                logger.info("Ladrón ha alcanzado la casilla más lejana. Objetivo cumplido.")
            break

        # Enviar comandos en intervalos
        if frame_counter % COMMAND_INTERVAL == 0:
            if command:
                comunicacionBluetooth.send_command(command)
                logger.info(f"Comando enviado en intervalo: {command}")

        # Esperar antes de la siguiente iteración
        time.sleep(0.5)  # Ajusta según la velocidad del robot

# ==============================
# Función para Capturar Frames
# ==============================

def capture_frames():
    """
    Captura continuamente frames de la cámara y actualiza la variable global latest_frame.
    """
    global latest_frame
    while True:
        ret, frame = camera.read()
        if ret:
            latest_frame = frame

# ==============================
# Funciones para Entrenar y Cargar Políticas
# ==============================

def train_policies_if_needed():
    """
    Entrena las políticas para Policía y Ladrón si no existen archivos de política.
    """
    policia_policy_file = 'policy_policia.pkl'
    ladron_policy_file = 'policy_ladron.pkl'

    # Verificar si las políticas ya están entrenadas
    if not os.path.exists(policia_policy_file) or not os.path.exists(ladron_policy_file):
        logger.info("Iniciando entrenamiento de políticas...")

        # Definir el mapa del laberinto
        maze = [
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ]
        rows = len(maze)
        cols = len(maze[0])

        # Definir objetivos
        policia_target = (4, 4)  # Casilla objetivo para Policía (puedes ajustarlo)
        ladron_target = (4, 4)    # Casilla objetivo para Ladrón (puedes ajustarlo)

        # Inicializar agentes
        agente_policia = QLearningAgent(rows, cols, role=0)
        agente_ladron = QLearningAgent(rows, cols, role=1)

        # Entrenar agente Policía
        logger.info("Entrenando agente Policía...")
        agente_policia.train(
            maze=maze,
            target_position=policia_target,
            episodes=1000
        )
        agente_policia.save_policy(policia_policy_file)

        # Entrenar agente Ladrón
        logger.info("Entrenando agente Ladrón...")
        # Supongamos que el Policía está en (0,0) durante el entrenamiento
        police_position = (0, 0)
        agente_ladron.train(
            maze=maze,
            target_position=ladron_target,
            police_position=police_position,
            episodes=1000
        )
        agente_ladron.save_policy(ladron_policy_file)

        logger.info("Entrenamiento de políticas completado.")
    else:
        logger.info("Políticas ya entrenadas. Cargando políticas existentes.")

def load_policies():
    """
    Carga las políticas de Q-learning para Policía y Ladrón desde archivos.
    """
    policia_policy_file = 'policy_policia.pkl'
    ladron_policy_file = 'policy_ladron.pkl'

    agente_policia = QLearningAgent(rows, cols, role=0)
    agente_ladron = QLearningAgent(rows, cols, role=1)

    agente_policia.load_policy(policia_policy_file)
    agente_ladron.load_policy(ladron_policy_file)

    return agente_policia, agente_ladron

# ==============================
# Funciones de Flask para Detectar y Procesar Shapes
# ==============================

@app.route('/detect_shapes', methods=['GET'])
def detect_shapes_endpoint():
    global latest_frame
    if latest_frame is not None:
        # Copiar el frame para no modificar la referencia global
        frame_copy = latest_frame.copy()
        # Dibujar la grilla y detectar formas
        shapes = detect_shapes_in_image(frame_copy, rows, cols)
        shapes = validate_and_convert_dict(shapes)
        return jsonify(shapes)
    else:
        return jsonify({"error": "No frame available from the camera"}), 500

# ==============================
# Bloque Principal
# ==============================

if __name__ == '__main__':
    try:
        # Entrenar las políticas si es necesario
        train_policies_if_needed()

        # Cargar las políticas entrenadas
        agente_policia, agente_ladron = load_policies()

        # Iniciar el hilo de captura de frames
        capture_thread = threading.Thread(target=capture_frames, daemon=True)
        capture_thread.start()
        logger.info("Hilo de captura de frames iniciado.")

        # Iniciar el hilo de control del robot
        controlador_thread = threading.Thread(target=control_robot, args=(agente_policia, agente_ladron), daemon=True)
        controlador_thread.start()
        logger.info("Hilo de controlador del robot iniciado.")

        # Iniciar el servidor Flask
        app.run(host='0.0.0.0', port=5000)
    finally:
        camera.release()  # Liberar la cámara al cerrar el servidor
        comunicacionBluetooth.close_connection()
        logger.info("Recursos liberados y conexión Bluetooth cerrada.")
