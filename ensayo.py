# analisisMapa_refactorizado_completo.py

import cv2
import numpy as np
import random
import threading
import queue
import math
import time
import logging
import comunicacionArduino  # Asegúrate de que el nombre del archivo sea correcto

# Configuración de Logging
logging.basicConfig(
    level=logging.INFO,  # Cambia a DEBUG para más detalles
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# Configuración de la Fuente de Video
# ==============================

SOURCE_URL = True  # True para usar DroidCam, False para cámara local
URL = "http://192.168.37.118:4747/video"
CAMERA_INDEX = 0

# Resolución reducida para mejorar el rendimiento
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ==============================
# Parámetros de la Cuadrícula y Canny
# ==============================

rows = 4
cols = 4
thickness = 1

canny_threshold1 = 50
canny_threshold2 = 150

# ==============================
# Control del Programa
# ==============================

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

    def stop(self):
        self.running = False

# ==============================
# Clase para Controlar el Estado del Robot
# ==============================

class RobotController:
    def __init__(self, maze, qr_detector):
        self.maze = maze
        self.qr_detector = qr_detector
        self.current_row = None
        self.current_col = None
        self.current_angle = None
        self.orientation_aligned = False
        self.movement_started = False
        self.target_row = 0
        self.target_col = 0
        self.margen = 30  # Grados de tolerancia para la alineación

    def update_position_and_angle(self, qr_data):
        """
        Actualiza la posición y ángulo actual del robot basado en los datos del QR.
        """
        self.current_row = qr_data['row']
        self.current_col = qr_data['col']
        self.current_angle = qr_data['angle']
        logger.info(f"Posición actual detectada: fila {self.current_row}, columna {self.current_col}")
        logger.info(f"Ángulo actual detectado: {self.current_angle:.2f}°")

    def needs_alignment(self):
        """
        Determina si el robot necesita alinear su orientación.
        """
        if self.current_angle is None:
            return False
        # Definimos un margen de tolerancia (por ejemplo 5 grados)
        margen = 5
        # Verificamos si el ángulo está fuera del rango de [90 - margen, 90 + margen]
        return not (90 - margen <= self.current_angle <= 90 + margen)

    def align_orientation(self):
        """
        Alinea la orientación del robot hacia 90 grados.
        """
        logger.info("Alineando a la derecha...")
        for _ in range(10):
            turn_right()
        time.sleep(0.5)  # Espera a que el robot gire
        self.orientation_aligned = True  # Asume que se ha alineado correctamente

    def move_forward_until_next_cell(self):
        """
        Mueve el robot hacia adelante hasta que detecta que ha pasado a la siguiente celda.
        """
        logger.info("Moviéndose hacia adelante...")
        move_forward()
        time.sleep(0.5)  # Espera a que el robot se mueva
        # La detección de la nueva celda se maneja en el bucle principal

    def decide_next_move(self):
        """
        Decide la siguiente acción basada en la posición actual y el objetivo.
        """
        if self.current_row == self.target_row and self.current_col == self.target_col:
            logger.info("Robot ha llegado a la posición objetivo (0,0).")
            return "DONE"

        # Decidir si mover hacia arriba o hacia la izquierda
        if self.current_row > self.target_row:
            desired_direction = "UP"
        elif self.current_col > self.target_col:
            desired_direction = "LEFT"
        else:
            desired_direction = "DONE"

        return desired_direction

    def execute_move(self, direction):
        """
        Ejecuta el movimiento basado en la dirección deseada.
        """
        if direction == "UP":
            logger.info("Preparándose para moverse hacia arriba.")
            if not self.needs_alignment():
                self.move_forward_until_next_cell()
            else:
                self.align_orientation()
        elif direction == "LEFT":
            logger.info("Preparándose para moverse hacia la izquierda.")
            if not self.needs_alignment():
                self.move_forward_until_next_cell()
            else:
                self.align_orientation()
        elif direction == "DONE":
            logger.info("Movimiento completado.")
            return "DONE"
        return "CONTINUE"

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

    if points is not None:
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
    comunicacionArduino.send_command('w')

def move_back():
    comunicacionArduino.send_command('s')

def turn_left():
    comunicacionArduino.send_command('a')

def turn_right():
    comunicacionArduino.send_command('d')

# ==============================
# Función Principal
# ==============================

def main():
    global running

    # Abrir la fuente de video según la configuración
    if SOURCE_URL:
        cap = cv2.VideoCapture(URL)
        logger.info(f"Usando la URL de DroidCam: {URL}")
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        logger.info(f"Usando la cámara incorporada del computador (Índice: {CAMERA_INDEX})")

    # Establecer una resolución más baja para mejorar el rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        logger.error("No se pudo conectar a la cámara seleccionada.")
        return

    logger.info(f"Conexión exitosa. Analizando video con cuadrícula de {rows}x{cols}...")
    maze = maze_generate(rows, cols)
    logger.debug("Mapa generado:")
    for fila in maze:
        logger.debug(fila)

    # Crear ventana y trackbars
    cv2.namedWindow('Ajustes')
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

    try:
        contador=0
        while running:
            if not frame_queue.empty():
                frame = frame_queue.get()
            else:
                continue

            # Obtener valores de las trackbars
            threshold1 = cv2.getTrackbarPos('Canny Th1', 'Ajustes')
            threshold2 = cv2.getTrackbarPos('Canny Th2', 'Ajustes')
            dilatacion = cv2.getTrackbarPos('Dilatacion', 'Ajustes')

            # Procesar el frame
            detected_qrs, frame_with_shapes = detect_qr_in_image(frame, rows, cols, qr_detector)
            frame_with_grid = draw_grid(frame_with_shapes, rows, cols, thickness)
            frame_filled = fill_cells(frame_with_grid, maze)
            frame_highlighted = highlight_start_end(frame_filled, rows, cols)

            # Mostrar el frame con los ajustes
            cv2.imshow('Cuadrícula con análisis', frame_highlighted)

            if contador % 50 == 0:

                # Procesar los QR detectados
                for qr in detected_qrs:
                    logger.debug(qr)
                    if robot.current_row is None or robot.current_col is None:
                        # Establecer la posición inicial basada en el QR detectado
                        robot.update_position_and_angle(qr)
                    else:
                        # Actualizar posición y ángulo
                        robot.update_position_and_angle(qr)

                    # Decidir el siguiente movimiento
                    next_move = robot.decide_next_move()
                    if next_move == "DONE":
                        logger.info("Robot ha alcanzado el objetivo (0,0).")
                        running = False
                        comunicacionArduino.send_command('q')
                        break
                    elif not robot.orientation_aligned:
                        if robot.needs_alignment():
                            robot.align_orientation()
                    elif not robot.movement_started:
                        robot.execute_move(next_move)
            # Incrementar el contador
            contador += 1

            # Presiona 'q' en la ventana de video para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Presionaste 'q'. Cerrando conexión y terminando programa...")
                running = False
                comunicacionArduino.send_command('q')
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

        # Cerrar la conexión serial si está abierta
        comunicacionArduino.close_connection()

if __name__ == "__main__":
    main()
