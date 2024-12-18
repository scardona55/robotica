import cv2
import numpy as np
import random
import math
import time
import logging
import json
import comunicacionBluetooth  # Asegúrate de que este módulo esté correctamente implementado
from collections import deque
from gridworld_utils import obtener_mapa_descriptivo, obtener_salida, obtener_obstaculos
import qlearn  # Importamos nuestro módulo de Q-learning

# ==============================
# Configuración del Logging
# ==============================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# Parámetros de la Cámara y Resolución
# ==============================

URL = "http://192.168.37.118:4747/video"  # Cambia esta URL según tu configuración
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ==============================
# Parámetros de la Cuadrícula
# ==============================

ROWS = 4
COLS = 4
THICKNESS = 1

CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# Intervalo para enviar comandos al Arduino (en frames)
COMMAND_INTERVAL = 48

# ==============================
# Clase para Controlar el Estado del Robot
# ==============================

class RobotController:
    def __init__(self, maze, qr_detector):
        self.maze = maze
        self.qr_detector = qr_detector
        self.rows = len(maze)
        self.cols = len(maze[0])
        
        # Información de posición y navegación
        self.current_row = None
        self.current_col = None
        self.current_angle = None
        self.target = obtener_salida(obtener_mapa_descriptivo(maze))  # Salida
        
        # Estado de navegación
        self.path = []  
        self.intentos_alineacion = 0
        self.movimiento_en_curso = False

    def calcular_camino_optimo(self):
        if self.current_row is None or self.current_col is None:
            logger.error("Posición actual no definida")
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
        logger.error("No se encontró un camino al objetivo")
        return []

    def ajustar_angulo(self, objetivo_angulo):
        margen_tolerancia = 20
        diff = (objetivo_angulo - self.current_angle + 360) % 360

        # Verificar si el ángulo ya está alineado
        if diff <= margen_tolerancia or diff >= 360 - margen_tolerancia:
            logger.info("Ángulo ya alineado.")
            return self.current_angle

        # Determinar la dirección óptima del giro
        if diff > 180:
            logger.info("Giro a la izquierda.")
            qlearn.turn_left()
            # Esperar a que el giro se complete
            time.sleep(1)  # Ajusta este tiempo según la velocidad de giro de tu robot
            self.current_angle = (self.current_angle - 90) % 360
        else:
            logger.info("Giro a la derecha.")
            qlearn.turn_right()
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
            logger.warning("No se detectó ningún QR.")
            return False

        # Actualizar posición y ángulo basados en el primer QR detectado
        qr_data = detected_qrs[0]  # Asumimos que el primer QR es el relevante
        self.update_position_and_angle(qr_data)

        # Comprobar si ha llegado a la casilla de destino
        if self.current_row == self.target[0] and self.current_col == self.target[1]:
            logger.info(f"El robot ha llegado a la posición objetivo: {self.target}.")
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
        logger.info(f"Ejecutando movimiento hacia: {direccion}")
        if direccion == "ARRIBA":
            self.ajustar_angulo(90)
            qlearn.move_forward()
        elif direccion == "ABAJO":
            self.ajustar_angulo(270)
            qlearn.move_forward()
        elif direccion == "IZQUIERDA":
            self.ajustar_angulo(180)
            qlearn.turn_left()
        elif direccion == "DERECHA":
            self.ajustar_angulo(0)
            qlearn.turn_right()
        
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
        
        logger.info(f"Posición actual: fila {self.current_row}, columna {self.current_col}")
        logger.info(f"Ángulo actual: {self.current_angle:.2f}°")

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

def draw_grid(frame, rows, cols, thickness=1):
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    for i in range(1, rows):
        cv2.line(frame, (0, i * cell_height), (width, i * cell_height), (0, 255, 0), thickness)
    for j in range(1, cols):
        cv2.line(frame, (j * cell_width, 0), (j * cell_width, height), (0, 255, 0), thickness)
    return frame

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

# Las funciones de movimiento ahora están en qlearn.py y se llaman desde allí.
# Por lo tanto, no es necesario redefinirlas aquí.
# Si prefieres mantenerlas en main.py, simplemente mueve las funciones de qlearn.py a main.py y ajústalas.

# ==============================
# Función Principal
# ==============================

def main():
    global ROWS, COLS, CANNY_THRESHOLD1, CANNY_THRESHOLD2

    # Abrir la fuente de video según la configuración
    cap = cv2.VideoCapture(URL)
    logger.info(f"Usando la URL de DroidCam: {URL}")

    # Establecer una resolución más baja para mejorar el rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        logger.error("No se pudo conectar a la cámara seleccionada.")
        return

    logger.info(f"Conexión exitosa. Analizando video con cuadrícula de {ROWS}x{COLS}...")
    maze = maze_generate(ROWS, COLS)
    logger.debug("Mapa generado:")
    for fila in maze:
        logger.debug(fila)

    mapa = obtener_mapa_descriptivo(maze)
    salida = obtener_salida(mapa)
    obstaculos = obtener_obstaculos(mapa)

    # Crear ventana y trackbars
    cv2.namedWindow('Ajustes')
    cv2.createTrackbar('Canny Th1', 'Ajustes', CANNY_THRESHOLD1, 255, lambda x: None)
    cv2.createTrackbar('Canny Th2', 'Ajustes', CANNY_THRESHOLD2, 255, lambda x: None)
    cv2.createTrackbar('Dilatacion', 'Ajustes', 2, 15, lambda x: None)

    qr_detector = cv2.QRCodeDetector()

    # Crear el controlador del robot
    robot = RobotController(maze, qr_detector)

    # Inicializar contador para el envío de comandos
    frame_counter = 0

    # Variables para Q-learning
    Q = qlearn.inicializar_Q(mapa)
    politica = {}
    policy_generated = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Error al capturar el video.")
                break

            frame_counter += 1

            # Obtener los valores actuales de los trackbars
            CANNY_THRESHOLD1 = cv2.getTrackbarPos('Canny Th1', 'Ajustes')
            CANNY_THRESHOLD2 = cv2.getTrackbarPos('Canny Th2', 'Ajustes')
            dilatacion = cv2.getTrackbarPos('Dilatacion', 'Ajustes')

            # Aplicar Canny (si es necesario)
            edges = cv2.Canny(frame, CANNY_THRESHOLD1, CANNY_THRESHOLD2)
            # Aquí podrías aplicar más procesamiento si es necesario

            # Verificar posición basada en el QR detectado
            if robot.verificar_posicion(frame):
                logger.info("El robot ha alcanzado la posición objetivo.")
                break

            # Procesar el frame (solo para visualización)
            _, frame_with_shapes = detect_qr_in_image(frame, ROWS, COLS, qr_detector)
            frame_with_grid = draw_grid(frame_with_shapes, ROWS, COLS, THICKNESS)
            frame_filled = fill_cells(frame_with_grid, maze)
            frame_highlighted = highlight_start_end(frame_filled, ROWS, COLS)

            # Mostrar el frame con los ajustes
            cv2.imshow('Cuadrícula con análisis', frame_highlighted)

            # Decidir y ejecutar el siguiente movimiento cada COMMAND_INTERVAL frames
            if frame_counter % COMMAND_INTERVAL == 0 and not policy_generated:
                # Verificar si el robot está en la posición inicial
                if (robot.current_row, robot.current_col) == (0, 0):
                    logger.info("Robot en posición inicial. Iniciando Q-learning...")
                    print("Calculando política...")
                    Q, politica = qlearn.entrenar_Q_learning(
                        maze=maze,
                        Q=Q,
                        salida=salida,
                        alpha=0.1,    # Puedes ajustar estos parámetros si lo deseas
                        gamma=0.9,
                        epsilon=1.0,
                        min_epsilon=0.1,
                        decay_rate=0.995,
                        episodes=1000   # Puedes ajustar el número de episodios
                    )
                    policy_generated = True
                    print("Política calculada. Ejecutando movimientos según la política...")
                else:
                    # Si no está en la posición inicial, buscar volver a ella
                    logger.info("El robot no está en la posición inicial. Volviendo a la posición inicial...")
                    direccion = robot.decidir_movimiento()
                    if direccion:
                        resultado = robot.ejecutar_movimiento(direccion)
                        if resultado == "TERMINADO":
                            logger.info("El robot ha alcanzado la posición inicial (0,0).")

            elif policy_generated:
                # Ejecutar la política
                estado_actual = (robot.current_row, robot.current_col)
                if estado_actual in politica:
                    accion = politica[estado_actual]
                    logger.info(f"Acción según la política en {estado_actual}: {accion}")
                    # Ejecutar la acción
                    if accion == 'up':
                        robot.ajustar_angulo(90)
                        qlearn.move_forward()
                    elif accion == 'down':
                        robot.ajustar_angulo(270)
                        qlearn.move_forward()
                    elif accion == 'left':
                        robot.ajustar_angulo(180)
                        qlearn.turn_left()
                    elif accion == 'right':
                        robot.ajustar_angulo(0)
                        qlearn.turn_right()
                    
                    # Después de ejecutar la acción, esperar a que el robot actualice su posición
                    time.sleep(1)  # Ajusta este tiempo según la velocidad de movimiento de tu robot
                else:
                    logger.warning(f"No hay acción definida en la política para la posición {estado_actual}.")

            # Presiona 'q' en la ventana de video para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Presionaste 'q'. Cerrando conexión y terminando programa...")
                break

    except KeyboardInterrupt:
        logger.info("Interrupción por teclado. Cerrando programa...")

    finally:
        # Libera recursos
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
