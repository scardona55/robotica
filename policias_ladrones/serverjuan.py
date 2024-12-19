from flask import Flask, jsonify, request, Response
import cv2
import numpy as np
import math
import threading
import queue
import time
import logging
import json  # Para cargar la política
import comunicacionBluetooth  # Asegúrate de que este módulo esté correctamente configurado
from collections import deque

app = Flask(__name__)

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
URL = "http://192.168.37.118:4747/video"  # URL de la cámara IP
camera = cv2.VideoCapture(URL)  # Abrir la cámara
# Si prefieres usar una cámara local, descomenta la siguiente línea y comenta la anterior
# camera = cv2.VideoCapture(0)

latest_frame = None
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()

# Parámetros de la cuadrícula
rows = 5
cols = 5
maze = [
    [0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
]

# Roles de los marcadores ArUco
roles = {
    8: 0,  # El valor 8 será el policía
    9: 1   # El valor 9 será el ladrón
}
bandera = 0  # Variable de control para alternar roles

# Variable para controlar el estado de ejecución del robot
robot_running = True

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
# Clase para Controlar el Estado del Robot
# ==============================

class RobotController:
    def __init__(self, maze, qr_detector, policy):
        self.maze = maze
        self.qr_detector = qr_detector
        self.policy = policy  # Política cargada
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
        detected_qrs, _ = detect_qr_in_image(image, self.rows, self.cols, self.qr_detector)
        
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

            # Obtener la mejor acción de la política
            best_action = self.policy.get(str(current_state), "OBJETIVO_ALCANZADO")

            if best_action == "ARRIBA":
                direccion = "ARRIBA"
            elif best_action == "ABAJO":
                direccion = "ABAJO"
            elif best_action == "IZQUIERDA":
                direccion = "IZQUIERDA"
            elif best_action == "DERECHA":
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

def maze_generate(filas, columnas):
    laberinto = [[1 for _ in range(columnas)] for _ in range(filas)]
    direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def en_rango(x, y):
        return 0 <= x < filas and 0 <= y < columnas

    def dfs(x, y):
        laberinto[x][y] = 0
        random.shuffle(direcciones)
        for dx, dy in direcciones:
            nx, ny = x + 2 * dx, y + 2 * dy
            if en_rango(nx, ny) and laberinto[nx][ny] == 1:
                laberinto[x + dx][y + dy] = 0
                dfs(nx, ny)

    laberinto[0][0] = 0
    dfs(0, 0)
    laberinto[filas - 1][columnas - 1] = 0

    # Conectar la salida al camino más cercano si está aislada
    if laberinto[filas - 2][columnas - 1] == 1 and laberinto[filas - 1][columnas - 2] == 1:
        laberinto[filas - 2][columnas - 1] = 0

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

def calculate_angle(points):
    top_left = points[0]
    top_right = points[1]
    delta_y = top_right[1] - top_left[1]
    delta_x = top_right[0] - top_left[0]
    angle = np.arctan2(delta_y, delta_x)
    return np.degrees(angle)

def normalize_angle(angle):
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle

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
# Funciones de Movimiento del Robot
# ==============================

def move_forward():
    for i in range(0,10):
        comunicacionBluetooth.send_command('w')
    print("Movimiento hacia adelante ejecutado.")

def move_back():
    for i in range(0,10):
        comunicacionBluetooth.send_command('s')
    print("Movimiento hacia atrás ejecutado.")

def turn_left():
    for i in range(0, 40):
        comunicacionBluetooth.send_command('d')  # Envía el comando
        # time.sleep(0.1)  # Descomenta si necesitas pausas
    print("Giro a la izquierda ejecutado.")

def turn_right():
    for i in range(0, 40):
        comunicacionBluetooth.send_command('a')  # Envía el comando
        # time.sleep(0.1)  # Descomenta si necesitas pausas
    print("Giro a la derecha ejecutado.")

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
# Rutas de la API
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

    logger.info(f"Roles actualizados: {roles}")

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
                detected_shapes = detect_shapes_in_image(frame_copy, rows, cols, aruco_dict, parameters)
                draw_grid(frame_copy, rows, cols)
                fill_cells(frame_copy, maze)
                highlight_start_end(frame_copy, rows, cols)
                # Codificar la imagen como JPEG
                _, buffer = cv2.imencode('.jpg', frame_copy)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

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
        shapes = detect_shapes_in_image(frame_copy, rows, cols, aruco_dict, parameters)
        shapes = validate_and_convert_dict(shapes)
        return jsonify(shapes)
    else:
        return jsonify({"error": "No frame available from the camera"}), 500

# ==============================
# Función de Detección de Marcadores ArUco
# ==============================

def detect_shapes_in_image(image, rows, cols, aruco_dict, parameters):
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
            row = int(math.floor(center[1]) // cell_height)
            col = int(math.floor(center[0]) // cell_width)
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
            roleTxt = "Policia" if role == 0 else "Ladron"
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
# Función Principal
# ==============================

def main():
    global latest_frame, robot_running

    # Cargar la política desde 'policy.json'
    try:
        with open('policy.json', 'r', encoding='utf-8') as f:
            policy_data = json.load(f)
    except FileNotFoundError:
        logger.error("El archivo 'policy.json' no se encontró. Asegúrate de generarlo antes de ejecutar el programa.")
        return
    except json.JSONDecodeError:
        logger.error("Error al decodificar 'policy.json'. Asegúrate de que el archivo esté en formato JSON válido.")
        return

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

    logger.info("Política cargada exitosamente.")

    # Generar o cargar el laberinto
    # Puedes usar una función de generación si lo prefieres
    # maze = maze_generate(rows, cols)  # Descomenta si quieres generar un nuevo laberinto
    # En este ejemplo, usamos el laberinto predefinido
    logger.info("Laberinto cargado:")
    for fila in maze:
        logger.info(fila)

    qr_detector = cv2.QRCodeDetector()
    robot = RobotController(maze, qr_detector, policy_dict)

    # Iniciar el FrameGrabber
    frame_queue = queue.Queue(maxsize=50)
    frame_grabber = FrameGrabber(camera, frame_queue)
    frame_grabber.start()

    try:
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                latest_frame = frame  # Actualizar el frame más reciente

                # Procesar el frame para detección y visualización
                detected_shapes = detect_shapes_in_image(frame, rows, cols, aruco_dict, parameters)
                draw_grid(frame, rows, cols)
                fill_cells(frame, maze)
                highlight_start_end(frame, rows, cols)

                # Mostrar el frame en una ventana
                cv2.imshow('Cuadrícula con análisis', frame)

                # Actualizar la posición del robot si hay marcadores detectados
                if detected_shapes:
                    robot.verificar_posicion(frame)

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

        # Libera recursos
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
