import cv2
import numpy as np
import random
import threading
import queue
import math
import time
import logging
from collections import deque

# Asegúrate de que gridworld_utils contiene las funciones necesarias:
# initialize_environment, initialize_q_table, train_agent, extract_policy
from gridworld_utils import initialize_environment, initialize_q_table, train_agent, extract_policy

# Importa tu módulo de comunicación Bluetooth
import comunicacionBluetooth  # Asegúrate de que este módulo funcione correctamente

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# URL de la cámara DroidCam
URL = "http://192.168.37.118:4747/video"

# Resolución reducida para mejorar el rendimiento
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Parámetros de la Cuadrícula 
rows = 4
cols = 4
thickness = 1

canny_threshold1 = 50
canny_threshold2 = 150

# Control del Programa
running = True

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
                logger.debug("Frame capturado y agregado a la cola.")
            else:
                logger.warning("La cola de frames está llena. Frame descartado.")
            # Opcional: Limitar la tasa de captura para evitar sobrecarga
            time.sleep(0.01)  # Ajusta según sea necesario

    def stop(self):
        self.running = False

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
        self.current_row = 0  # Asumiendo que inicia en (0,0)
        self.current_col = 0
        self.current_angle = 0  # 0 grados hacia la derecha
        
        # Casilla objetivo (puedes ajustar esto según tu lógica)
        self.target_row = len(maze) - 1
        self.target_col = len(maze[0]) - 1

    def actualizar_posicion(self, qr_data):
        """
        Actualiza la posición y ángulo del robot basándose en los datos del QR.
        """
        self.current_row = qr_data['row']
        self.current_col = qr_data['col']
        self.current_angle = qr_data['angle']
        
        logger.info(f"Posición actual: fila {self.current_row}, columna {self.current_col}")
        logger.info(f"Ángulo actual: {self.current_angle:.2f}°")

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
        self.actualizar_posicion(qr_data)

        # Comprobar si ha llegado a la casilla de destino
        if self.current_row == self.target_row and self.current_col == self.target_col:
            logger.info(f"El robot ha llegado a la posición objetivo: ({self.target_row}, {self.target_col}).")
            return True

        return False

# ==============================
# Funciones de Generación y Descripción del Mapa
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
    if filas > 1 and columnas > 1:
        if laberinto[filas - 2][columnas - 1] == 1 and laberinto[filas - 1][columnas - 2] == 1:
            laberinto[filas - 2][columnas - 1] = 0

    return laberinto

def obtener_mapa_descriptivo(maze):
    """
    Genera una matriz descriptiva del estado del mapa.
    'S' representa el inicio.
    'E' representa la salida.
    'O' representa obstáculos.
    'P' representa pasillos.
    """
    rows = len(maze)
    cols = len(maze[0])
    mapa_descriptivo = []

    for i in range(rows):
        fila = []
        for j in range(cols):
            if i == 0 and j == 0:  # Inicio
                fila.append('S')
            elif i == rows - 1 and j == cols - 1:  # Salida
                fila.append('E')
            elif maze[i][j] == 1:  # Obstáculo
                fila.append('O')
            else:  # Pasillo
                fila.append('P')
        mapa_descriptivo.append(fila)

    return mapa_descriptivo

def obtener_salida(mapa):
    """
    Encuentra la posición de la salida 'E' en el mapa.
    
    Parámetros:
    mapa (list of list): Matriz representando el mapa.
    
    Retorna:
    tuple: Coordenadas de la salida (fila, columna).
    """
    for i, fila in enumerate(mapa):
        for j, celda in enumerate(fila):
            if celda == 'E':
                return (i, j)
    return None  # Devuelve None si no encuentra la salida

def obtener_obstaculos(mapa):
    """
    Encuentra las posiciones de los obstáculos 'O' en el mapa.
    
    Parámetros:
    mapa (list of list): Matriz representando el mapa.
    
    Retorna:
    list: Lista de coordenadas de los obstáculos [(fila1, columna1), (fila2, columna2), ...].
    """
    obstaculos = []
    for i, fila in enumerate(mapa):
        for j, celda in enumerate(fila):
            if celda == 'O':
                obstaculos.append((i, j))
    return obstaculos

# ==============================
# Funciones de Dibujo y Procesamiento de Frames
# ==============================

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

def move_forward():
    for _ in range(10):
        comunicacionBluetooth.send_command('w')
    logger.info("Movimiento hacia adelante ejecutado.")

def move_back():
    for _ in range(10):
        comunicacionBluetooth.send_command('s')
    logger.info("Movimiento hacia atrás ejecutado.")

def turn_left():
    for _ in range(40):
        comunicacionBluetooth.send_command('a')  # 'a' para girar a la izquierda
        # time.sleep(0.1)  # Opcional: espera entre comandos
    logger.info("Giro a la izquierda ejecutado.")

def turn_right():
    for _ in range(40):
        comunicacionBluetooth.send_command('d')  # 'd' para girar a la derecha
        # time.sleep(0.1)  # Opcional: espera entre comandos
    logger.info("Giro a la derecha ejecutado.")

# ==============================
# Función para Ejecutar la Política
# ==============================

def ejecutar_politica(politica, posicion_actual, robot):
    """
    Ejecuta la acción correspondiente según la política y la posición actual.

    Parámetros:
    politica (dict): Diccionario con la política, donde las claves son las posiciones
                     en formato string "(fila, columna)",
                     y los valores son las acciones ('up', 'down', 'left', 'right').
    posicion_actual (tuple): Tupla que indica la posición actual del robot (fila, columna).
    robot (RobotController): Instancia del controlador del robot.

    Retorna:
    None
    """
    # Convertir la posición actual a string para buscar en la política
    posicion_str = str(posicion_actual)
    accion = politica.get(posicion_str)

    if accion is None:
        logger.warning(f"No hay acción definida para la posición {posicion_actual}.")
        return

    logger.info(f"Acción a ejecutar en posición {posicion_actual}: {accion}")

    # Determinar la orientación deseada basada en la acción
    # Suponiendo que 'up' = 90°, 'right' = 0°, 'down' = 270°, 'left' = 180°
    accion_angulos = {
        'up': 90,
        'right': 0,
        'down': 270,
        'left': 180
    }

    objetivo_angulo = accion_angulos.get(accion)

    if objetivo_angulo is None:
        logger.warning(f"Acción desconocida: {accion}")
        return

    # Calcular la diferencia de ángulo
    margen_tolerancia = 20  # Puedes ajustar este valor
    diff = (objetivo_angulo - robot.current_angle + 360) % 360

    if diff <= margen_tolerancia or diff >= 360 - margen_tolerancia:
        # Ya está orientado correctamente, mover hacia adelante
        move_forward()
    else:
        # Determinar la dirección del giro
        if diff > 180:
            # Girar a la izquierda
            turn_left()
            robot.current_angle = (robot.current_angle - 90) % 360
        else:
            # Girar a la derecha
            turn_right()
            robot.current_angle = (robot.current_angle + 90) % 360

# ==============================
# Función Principal
# ==============================

def main():
    global running

    while running:
        # Abrir la fuente de video según la configuración
        cap = cv2.VideoCapture(URL)
        logger.info(f"Usando la URL de DroidCam: {URL}")

        # Establecer una resolución más baja para mejorar el rendimiento
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not cap.isOpened():
            logger.error("No se pudo conectar a la cámara seleccionada.")
            time.sleep(5)  # Esperar antes de intentar reconectar
            continue

        logger.info(f"Conexión exitosa. Generando y analizando video con cuadrícula de {rows}x{cols}...")
        maze = maze_generate(rows, cols)
        logger.debug("Mapa generado:")
        for fila in maze:
            logger.debug(fila)

        # Crear ventana y trackbars
        cv2.namedWindow('Ajustes', cv2.WINDOW_NORMAL)
        cv2.createTrackbar('Canny Th1', 'Ajustes', canny_threshold1, 255, lambda x: None)
        cv2.createTrackbar('Canny Th2', 'Ajustes', canny_threshold2, 255, lambda x: None)
        cv2.createTrackbar('Dilatacion', 'Ajustes', 2, 15, lambda x: None)

        qr_detector = cv2.QRCodeDetector()

        # Crear el controlador del robot
        robot = RobotController(maze, qr_detector)

        # Iniciar el FrameGrabber
        frame_queue = queue.Queue(maxsize=50)
        frame_grabber = FrameGrabber(cap, frame_queue)
        frame_grabber.start()

        # Generar el mapa descriptivo y obtener la salida y obstáculos
        mapa_descriptivo = obtener_mapa_descriptivo(maze)
        salida = obtener_salida(mapa_descriptivo)
        obstaculos = obtener_obstaculos(mapa_descriptivo)
        height = len(mapa_descriptivo)  # Número de filas
        width = len(mapa_descriptivo[0]) if mapa_descriptivo else 0  # Número de columnas

        # Inicializar entorno
        env = initialize_environment(width, height, salida, obstaculos)

        # Parámetros de Q-learning
        alpha = 0.1
        gamma = 0.99
        epsilon = 1.0
        epsilon_min = 0.1
        epsilon_decay = 0.995
        episodes = 1000  # Puedes ajustar el número de episodios

        # Inicializar tabla Q
        Q = initialize_q_table(env.width, env.height, env.obstacles, env.actions)

        # Entrenar al agente
        logger.info("Entrenando al agente con Q-learning...")
        Q = train_agent(env, Q, episodes, alpha, gamma, epsilon, epsilon_min, epsilon_decay)
        logger.info("Entrenamiento completado.")

        # Extraer política
        policy = extract_policy(Q)
        logger.info("Política extraída.")

        try:
            while running:
                if not frame_queue.empty():
                    frame = frame_queue.get()
                else:
                    # Espera breve para evitar un bucle muy rápido si la cola está vacía
                    time.sleep(0.01)
                    continue

                # Procesar el frame (solo para visualización)
                detected_qrs, frame_with_shapes = detect_qr_in_image(frame, rows, cols, qr_detector)
                frame_with_grid = draw_grid(frame_with_shapes, rows, cols, thickness)
                frame_filled = fill_cells(frame_with_grid, maze)
                frame_highlighted = highlight_start_end(frame_filled, rows, cols)

                # Mostrar el frame con los ajustes
                cv2.imshow('Cuadrícula con análisis', frame_highlighted)

                # Procesar los QR detectados
                for qr in detected_qrs:
                    logger.debug(qr)
                    
                    # Actualizar posición y ángulo del robot
                    robot.actualizar_posicion(qr)

                    # Obtener la posición actual del robot
                    posicion_actual = (robot.current_row, robot.current_col)

                    # Si el robot está en la posición inicial (0,0), ejecutar la política
                    if posicion_actual == (0, 0):
                        ejecutar_politica(policy, posicion_actual, robot)
                    
                    # Verificar si el robot ha alcanzado la salida
                    if posicion_actual == salida:
                        logger.info("El robot ha alcanzado la salida. Finalizando programa.")
                        running = False
                        comunicacionBluetooth.send_command('q')  # Envía el comando de detener
                        break

                # Presiona 'q' en la ventana de video para salir
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Presionaste 'q'. Cerrando conexión y terminando programa...")
                    running = False
                    comunicacionBluetooth.send_command('q')  # Envía el comando de detener
                    break

        except KeyboardInterrupt:
            logger.info("Interrupción por teclado. Cerrando programa...")

        finally:
            # Detener el FrameGrabber
            frame_grabber.stop()
            frame_grabber.join()

            # Libera recursos
            cap.release()
            cv2.destroyAllWindows()

            # Reiniciar la cámara si aún está corriendo
            if running:
                logger.info("Reiniciando la cámara...")
                continue
            else:
                break

    if __name__ == "__main__":
        main()
