# server.py

from flask import Flask, jsonify, request, Response
import cv2
import numpy as np
import math
import threading
import time
import logging
import json
from collections import deque
import comunicacionBluetooth  # Asegúrate de que este módulo esté correctamente implementado
import qlearn  # Importamos nuestro módulo de Q-learning

app = Flask(__name__)

# Configuración del Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Inicializar variables globales para la cámara y la última imagen capturada
camera = cv2.VideoCapture("http://10.144.145.9:4747/video")  # Abrir la cámara IP
# camera = cv2.VideoCapture(0)  # Alternativa para cámara local
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

# Diccionario para almacenar controladores de robots
robot_controllers = {}  # key: marker_id, value: RobotController instance

# ==============================
# Clase RobotController Integrada
# ==============================

class RobotController:
    def __init__(self, marker_id, maze, agent_type=0):
        """
        Inicializa el controlador del robot.

        Parámetros:
            marker_id (int): ID del marcador ArUco que identifica al robot.
            maze (list of list of int): Mapa del laberinto.
            agent_type (int): Tipo de agente (0=Policía, 1=Ladrón).
        """
        self.marker_id = marker_id
        self.maze = maze
        self.agent_type = agent_type  # 0=Policía, 1=Ladrón
        self.qr_detector = cv2.QRCodeDetector()
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.target = obtener_salida(obtener_mapa_descriptivo(maze))  # Salida

        # Información de posición y navegación
        self.current_row = None
        self.current_col = None
        self.current_angle = None
        self.path = []
        self.intentos_alineacion = 0
        self.movimiento_en_curso = False

        # Comunicación Bluetooth (asegúrate de que este método esté implementado correctamente)
        self.bt_connection = comunicacionBluetooth.connect_robot(marker_id)

        # Estado de la política
        self.politica = {}
        self.policy_generated = False

        # Thread para controlar el robot
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

    def calcular_camino_optimo(self):
        if self.current_row is None or self.current_col is None:
            logger.error(f"Robot {self.marker_id}: Posición actual no definida")
            return []

        inicio = (self.current_row, self.current_col)
        objetivo = self.target

        queue_ = deque([(inicio, [])])
        visited = set([inicio])

        while queue_:
            (row, col), path = queue_.popleft()

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
                        queue_.append(((new_row, new_col), path + [(row, col)]))

        # Si no se encontró un camino, devolver una lista vacía
        logger.error(f"Robot {self.marker_id}: No se encontró un camino al objetivo")
        return []

    def ajustar_angulo(self, objetivo_angulo):
        margen_tolerancia = 20
        diff = (objetivo_angulo - self.current_angle + 360) % 360

        # Verificar si el ángulo ya está alineado
        if diff <= margen_tolerancia or diff >= 360 - margen_tolerancia:
            logger.info(f"Robot {self.marker_id}: Ángulo ya alineado.")
            return self.current_angle

        # Determinar la dirección óptima del giro
        if diff > 180:
            logger.info(f"Robot {self.marker_id}: Giro a la izquierda.")
            qlearn.turn_left(self.bt_connection)
            # Esperar a que el giro se complete
            time.sleep(1)  # Ajusta este tiempo según la velocidad de giro de tu robot
            self.current_angle = (self.current_angle - 90) % 360
        else:
            logger.info(f"Robot {self.marker_id}: Giro a la derecha.")
            qlearn.turn_right(self.bt_connection)
            # Esperar a que el giro se complete
            time.sleep(1)  # Ajusta este tiempo según la velocidad de giro de tu robot
            self.current_angle = (self.current_angle + 90) % 360

        return self.current_angle

    def verificar_posicion(self, image):
        """
        Verifica si el robot ha llegado a la casilla objetivo usando detect_qr_in_image.
        """
        # Detectar QR en la imagen capturada
        detected_qrs, _ = detect_qr_in_image(image, self.rows, self.cols, self.qr_detector)

        # Si no se detecta ningún QR, no actualizar posición
        if not detected_qrs:
            logger.warning(f"Robot {self.marker_id}: No se detectó ningún QR.")
            return False

        # Actualizar posición y ángulo basados en el primer QR detectado
        qr_data = detected_qrs[0]  # Asumimos que el primer QR es el relevante
        self.update_position_and_angle(qr_data)

        # Comprobar si ha llegado a la casilla de destino
        if self.current_row == self.target[0] and self.current_col == self.target[1]:
            logger.info(f"Robot {self.marker_id}: Ha llegado a la posición objetivo: {self.target}.")
            return True

        return False

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
        logger.info(f"Robot {self.marker_id}: Ejecutando movimiento hacia: {direccion}")
        if direccion == "ARRIBA":
            self.ajustar_angulo(90)
            qlearn.move_forward(self.bt_connection)
        elif direccion == "ABAJO":
            self.ajustar_angulo(270)
            qlearn.move_forward(self.bt_connection)
        elif direccion == "IZQUIERDA":
            self.ajustar_angulo(180)
            qlearn.turn_left(self.bt_connection)
        elif direccion == "DERECHA":
            self.ajustar_angulo(0)
            qlearn.turn_right(self.bt_connection)

        if self.path:
            self.path.pop(0)

        return "CONTINUAR" if self.path else "TERMINADO"

    def update_position_and_angle(self, qr_data):
        """
        Actualiza la posición y ángulo del robot
        """
        self.current_row = qr_data['row']
        self.current_col = qr_data['col']
        self.current_angle = qr_data['angle']

        logger.info(f"Robot {self.marker_id}: Posición actual: fila {self.current_row}, columna {self.current_col}")
        logger.info(f"Robot {self.marker_id}: Ángulo actual: {self.current_angle:.2f}°")

    def train_policy(self):
        """
        Entrena la política de Q-learning y la almacena.
        """
        logger.info(f"Robot {self.marker_id}: Iniciando entrenamiento de Q-learning...")
        policy_json = qlearn.train_and_get_policy(
            grid_size=(self.rows, self.cols),
            obstacles=obtener_obstaculos(obtener_mapa_descriptivo(self.maze)),
            num_episodes=1000,
            max_steps=100
        )

        # Convertir policy_json a un diccionario con claves (State, Agent_Type)
        # Formato: {(row, col, Agent_Type): 'action'}
        self.politica = {
            (tuple(entry['State']), entry['Agent_Type']): entry['Best_Action']
            for entry in policy_json
        }

        self.policy_generated = True
        logger.info(f"Robot {self.marker_id}: Política calculada y almacenada.")

    def control_loop(self):
        """
        Bucle de control principal para el robot.
        """
        try:
            while True:
                if not self.policy_generated:
                    # Esperar a que se asigne un rol y entrenar la política
                    time.sleep(1)
                    continue

                if (self.current_row, self.current_col) is None:
                    # Esperar a que se detecte la posición
                    time.sleep(1)
                    continue

                # Ejecutar movimientos según la política
                estado_actual = (self.current_row, self.current_col)
                accion = self.politica.get((estado_actual, self.agent_type), None)
                if accion:
                    logger.info(f"Robot {self.marker_id}: Acción según la política en {estado_actual} para Agent_Type {self.agent_type}: {accion}")
                    # Ejecutar la acción
                    if accion == 'up':
                        self.ajustar_angulo(90)
                        qlearn.move_forward(self.bt_connection)
                    elif accion == 'down':
                        self.ajustar_angulo(270)
                        qlearn.move_forward(self.bt_connection)
                    elif accion == 'left':
                        self.ajustar_angulo(180)
                        qlearn.turn_left(self.bt_connection)
                    elif accion == 'right':
                        self.ajustar_angulo(0)
                        qlearn.turn_right(self.bt_connection)

                    # Después de ejecutar la acción, esperar a que el robot actualice su posición
                    time.sleep(1)  # Ajusta este tiempo según la velocidad de movimiento de tu robot
                else:
                    logger.warning(f"Robot {self.marker_id}: No hay acción definida en la política para la posición {estado_actual} y Agent_Type {self.agent_type}.")

                time.sleep(0.1)  # Pequeña pausa para evitar sobrecargar el ciclo

        except Exception as e:
            logger.error(f"Robot {self.marker_id}: Error en el bucle de control: {e}")

# ==============================
# Función para calcular el ángulo de inclinación
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

# ==============================
# Función para dibujar flechas indicando orientación
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
# Función para dibujar la cuadrícula sobre la imagen
# ==============================

def draw_grid(image, rows, cols, grid_values):
    """
    Dibuja una grilla sobre la imagen dada según el número de filas y columnas.
    Colorea en rojo las casillas correspondientes a un valor de 1 en la matriz grid_values con un alpha de 0.7.
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

# ==============================
# Función para detectar y procesar marcadores ArUco
# ==============================

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
            cv2.putText(image, f"{int(angle)}''", (int(center[0] - 50), int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Determinar la posición en la cuadrícula
            row = int(math.floor(center[1]) // cell_height)
            col = int(math.floor(center[0]) // cell_width)
            cell_index = row * cols + col
            cell_center_x = int((col + 0.5) * cell_width)
            cell_center_y = int((row + 0.5) * cell_height)

            cv2.putText(image, f"{cell_index}", (int(center[0]-10), int(center[1]+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            # Asignar rol basado en el ID del marcador y la bandera
            role = -1
            if marker_id in roles:
                role = roles[marker_id]
            roleTxt = "Policia" if role == 0 else "Ladron" if role == 1 else "Desconocido"
            cv2.putText(image, f"{roleTxt}", (int(center[0] - 10), int(center[1] + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            detected_shapes.append({
                "marker_id": marker_id,
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

            # Asignar el rol al robot y crear controlador si no existe
            if marker_id in roles and marker_id not in robot_controllers:
                agent_type = roles[marker_id]
                robot_controller = RobotController(marker_id, maze, agent_type=agent_type)
                robot_controllers[marker_id] = robot_controller
                # Iniciar entrenamiento de política
                robot_controller.train_policy()

    print(detected_shapes)
    return detected_shapes

# ==============================
# Función para validar y convertir diccionarios a tipos serializables JSON
# ==============================

def validate_and_convert_dict(input_data):
    """
    Recorre los valores de un diccionario o lista de diccionarios y valida que sean serializables a JSON.
    Convierte los tipos numpy a tipos nativos de Python cuando sea necesario.
    """
    if isinstance(input_data, dict):
        for key, value in input_data.items():
            if isinstance(value, np.generic):
                input_data[key] = value.item()
            elif isinstance(value, (dict, list)):
                input_data[key] = validate_and_convert_dict(value)
    elif isinstance(input_data, list):
        for i, item in enumerate(input_data):
            if isinstance(item, np.generic):
                input_data[i] = item.item()
            elif isinstance(item, (dict, list)):
                input_data[i] = validate_and_convert_dict(item)

    return input_data

# ==============================
# Función para obtener la salida del laberinto
# (Implementa según tu lógica en gridworld_utils)
# ==============================

def obtener_salida(mapa_descriptivo):
    # Implementa esta función según tu módulo gridworld_utils
    return obtener_salida(mapa_descriptivo)

def obtener_obstaculos(mapa_descriptivo):
    # Implementa esta función según tu módulo gridworld_utils
    return obtener_obstaculos(mapa_descriptivo)

def obtener_mapa_descriptivo(maze):
    # Implementa esta función según tu módulo gridworld_utils
    return obtener_mapa_descriptivo(maze)

# ==============================
# Función para detectar y procesar QR en la imagen
# (Integrada desde main.py)
# ==============================

def detect_qr_in_image(image, rows, cols, qr_detector):
    detected_qrs = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data, points, _ = qr_detector.detectAndDecode(gray)

    if points is not None and data:
        points = points.reshape((-1, 2)).astype(int)
        for i in range(len(points)):
            cv2.line(image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 3)

        angle = calculate_angle(points)
        angle = normalize_angle(angle)

        qr_center_x = int(np.mean(points[:, 0]))
        qr_center_y = int(np.mean(points[:, 1]))
        qr_center = (qr_center_x, qr_center_y)

        height, width = image.shape[:2]
        cell_width = width / cols
        cell_height = height / rows

        row = int(qr_center_y // cell_height)
        col = int(qr_center_x // cell_width)

        cell_center_x = int((col + 0.5) * cell_width)
        cell_center_y = int((row + 0.5) * cell_height)
        cell_center = (cell_center_x, cell_center_y)

        arrow_tip_zero = (qr_center_x + 50, qr_center_y)
        cv2.arrowedLine(image, qr_center, arrow_tip_zero, (0, 0, 255), 2, tipLength=0.3)

        angle_rad = np.radians(angle)
        arrow_tip_blue = (int(qr_center_x + 100 * np.cos(angle_rad)), int(qr_center_y + 100 * np.sin(angle_rad)))
        cv2.arrowedLine(image, qr_center, arrow_tip_blue, (255, 0, 0), 2, tipLength=0.3)

        angle2 = 360 - angle

        cell_index = row * cols + col

        detected_qrs.append({
            "shape": data,
            "angle": angle2,
            "x": qr_center_x,
            "y": qr_center_y,
            "cell_center_x": cell_center_x,
            "cell_center_y": cell_center_y,
            "cell_index": cell_index,
            "row": row,
            "col": col
        })

        cv2.putText(
            image,
            f"{cell_index}",
            (qr_center_x - 10, qr_center_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        cv2.putText(
            image,
            f"{qr_center_x},{qr_center_y}",
            (qr_center_x - 30, qr_center_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
        cv2.putText(image, f"{angle2:.2f}°", (qr_center_x - 30, qr_center_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                    2, cv2.LINE_AA)

        image = draw_dotted_line_in_cell(image, cell_center_x, cell_center_y, cell_width, cell_height)

    return detected_qrs, image

def normalize_angle(angle):
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle

def draw_dotted_line_in_cell(image, cell_center_x, cell_center_y, cell_width, cell_height):
    cell_left = int(cell_center_x - cell_width // 2)
    cell_right = int(cell_center_x + cell_width // 2)
    cell_top = int(cell_center_y - cell_height // 2)
    cell_bottom = int(cell_center_y + cell_height // 2)

    for x in range(cell_left, cell_right, 10):
        cv2.line(image, (x, cell_center_y), (x + 5, cell_center_y), (0, 0, 255), 1)

    for y in range(cell_top, cell_bottom, 10):
        cv2.line(image, (cell_center_x, y), (cell_center_x, y + 5), (0, 0, 255), 1)
    return image

# ==============================
# Función para generar y resaltar el laberinto
# (Integrada desde main.py)
# ==============================

def fill_cells(frame, matrix, alpha=0.7):
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
# Función para capturar imágenes continuamente
# ==============================

def capture_frames():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if ret:
            latest_frame = frame

# Iniciar el hilo de captura de frames
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

# ==============================
# Ruta para cambiar roles
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

    # Actualizar los roles en los controladores existentes
    for marker_id, controller in robot_controllers.items():
        if marker_id in roles:
            controller.agent_type = roles[marker_id]
            logger.info(f"Robot {marker_id}: Rol actualizado a {'Policía' if roles[marker_id] == 0 else 'Ladrón'}")
            if not controller.policy_generated:
                controller.train_policy()

    return jsonify({
        'bandera': bandera,
        'roles': roles
    })

# ==============================
# Ruta para el flujo de video procesado
# ==============================

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
                highlight_start_end(frame_copy, rows, cols)
                # Codificar la imagen como JPEG
                _, buffer = cv2.imencode('.jpg', frame_copy)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.1)  # Pequeña pausa para evitar sobrecarga

    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

# ==============================
# Ruta para obtener la matriz del laberinto
# ==============================

@app.route('/maze', methods=['GET'])
def get_maze():
    return jsonify(maze)

# ==============================
# Ruta para detectar y obtener información de los marcadores
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
# Función Principal para Ejecutar el Servidor
# ==============================

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        camera.release()  # Liberar la cámara al cerrar el servidor
