# analisisMapa.py
import cv2
import numpy as np
import random
import threading
import comunicacionArduino  # Asegúrate de que el nombre del archivo sea correcto
import queue
import math
import time

# ==============================
# Configuración de la Fuente de Video
# ==============================

# Modo de fuente de video
# Establece SOURCE_URL en True para usar la URL de DroidCam
# Establece SOURCE_URL en False para usar la cámara incorporada del computador
SOURCE_URL = True  # Cambiar a True para DroidCam, False para cámara local

# URL de DroidCam
URL = "http://10.144.88.177:4747/video"

# Índice de la cámara incorporada (normalmente 0)
CAMERA_INDEX = 0

# ==============================
# Parámetros de la Cuadrícula y Canny
# ==============================

# Parámetros de la cuadrícula
rows = 7  # Número de filas
cols = 7  # Número de columnas
thickness = 1  # Grosor de las líneas

# Valores iniciales de Canny (puedes ajustar si es necesario)
canny_threshold1 = 50
canny_threshold2 = 150

# Variable de control para el bucle principal
running = True

# ==============================
# Funciones de Generación y Dibujo del Laberinto
# ==============================

def maze_generate(filas, columnas):
    """
    Genera un laberinto de dimensiones filas x columnas.
    Los caminos están representados por 0 y las paredes por 1.
    Garantiza que (0,0) es el inicio y (filas-1,columnas-1) es la meta con un camino solucionable.
    """
    # Crear una matriz llena de paredes (1)
    laberinto = [[1 for _ in range(columnas)] for _ in range(filas)]

    # Direcciones de movimiento: (dx, dy) para celdas ortogonales
    direcciones = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def en_rango(x, y):
        """Verifica si una celda está dentro del rango del laberinto."""
        return 0 <= x < filas and 0 <= y < columnas

    def dfs(x, y):
        """Algoritmo DFS para construir el laberinto."""
        laberinto[x][y] = 0  # Marca el camino actual como "camino"
        random.shuffle(direcciones)  # Aleatoriza el orden de las direcciones
        for dx, dy in direcciones:
            nx, ny = x + 2 * dx, y + 2 * dy  # Saltar una celda para garantizar paredes entre caminos
            if en_rango(nx, ny) and laberinto[nx][ny] == 1:  # Si es una celda válida y no visitada
                # Romper la pared entre la celda actual y la siguiente
                laberinto[x + dx][y + dy] = 0
                # Continuar el DFS desde la celda siguiente
                dfs(nx, ny)

    # Inicializar el laberinto
    laberinto[0][0] = 0  # Crear la entrada
    dfs(0, 0)

    # Crear la salida
    laberinto[filas - 1][columnas - 1] = 0  # Asegurar que el punto final sea siempre un camino

    # Conectar la salida al camino más cercano si está aislada
    if laberinto[filas - 2][columnas - 1] == 1 and laberinto[filas - 1][columnas - 2] == 1:
        laberinto[filas - 2][columnas - 1] = 0  # Romper la pared superior

    # Devolver la matriz del laberinto
    return laberinto

def draw_grid(frame, rows, cols, thickness=1):
    """Dibuja una cuadrícula en el frame."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    for i in range(1, rows):  # Líneas horizontales
        cv2.line(frame, (0, i * cell_height), (width, i * cell_height), (0, 255, 0), thickness)
    for j in range(1, cols):  # Líneas verticales
        cv2.line(frame, (j * cell_width, 0), (j * cell_width, height), (0, 255, 0), thickness)
    return frame

def calculate_angle(points):
    """
    Calcula el ángulo de inclinación en grados de un código QR dado.
    Se basa en las coordenadas de las esquinas.
    """
    # Extraer las coordenadas de las esquinas superiores izquierda y derecha
    top_left = points[0]
    top_right = points[1]

    # Calcular el ángulo en radianes
    delta_y = top_right[1] - top_left[1]
    delta_x = top_right[0] - top_left[0]
    angle = np.arctan2(delta_y, delta_x)  # Ángulo en radianes

    # Convertir a grados
    return np.degrees(angle)

def normalize_angle(angle):
    """
    Normaliza el ángulo para que esté entre 0° y 360°.
    El ángulo aumenta en sentido contrario a las manecillas del reloj.
    """
    angle = angle % 360  # Asegura que el ángulo esté dentro del rango [0, 360)
    if angle < 0:
        angle += 360  # Convertir a un ángulo positivo
    return angle

def detect_qr_in_image(image, rows, cols, qr_detector):
    """
    Detecta códigos QR en la imagen, calcula su ángulo de inclinación,
    determina su posición en la cuadrícula y dibuja información relevante.
    """
    detected_qrs = []

    # Detectar y decodificar un solo código QR
    data, points, _ = qr_detector.detectAndDecode(image)

    if points is not None:
        points = points.reshape((-1, 2)).astype(int)

        # Dibujar los recuadros alrededor del código QR
        for i in range(len(points)):
            cv2.line(image, tuple(points[i]), tuple(points[(i + 1) % len(points)]), (0, 255, 0), 3)

        # Calcular la inclinación
        angle = calculate_angle(points)

        # Normalizar el ángulo para que esté en el rango [0, 360]
        angle = normalize_angle(angle)

        # Calcular el centro del QR
        qr_center_x = int(np.mean(points[:, 0]))
        qr_center_y = int(np.mean(points[:, 1]))
        qr_center = (qr_center_x, qr_center_y)

        # Calcular la fila y columna de la cuadrícula
        height, width = image.shape[:2]
        cell_width = width / cols
        cell_height = height / rows

        # Calcular en qué celda (fila, columna) se encuentra el centro del QR
        row = int(qr_center_y // cell_height)
        col = int(qr_center_x // cell_width)

        # Calcular el centro de la celda
        cell_center_x = int((col + 0.5) * cell_width)
        cell_center_y = int((row + 0.5) * cell_height)
        cell_center = (cell_center_x, cell_center_y)

        # Flecha indicando cero grados (horizontal a la derecha) desde el centro
        arrow_tip_zero = (qr_center_x + 50, qr_center_y)  # Flecha hacia la derecha (0°)
        cv2.arrowedLine(image, qr_center, arrow_tip_zero, (0, 0, 255), 2, tipLength=0.3)

        # Flecha azul indicando el ángulo detectado
        angle_rad = np.radians(angle)
        arrow_tip_blue = (int(qr_center_x + 100 * np.cos(angle_rad)), int(qr_center_y + 100 * np.sin(angle_rad)))
        cv2.arrowedLine(image, qr_center, arrow_tip_blue, (255, 0, 0), 2, tipLength=0.3)

        # Mostrar los datos y la inclinación en pantalla
        if data:
            # Puedes descomentar la siguiente línea para mostrar el contenido del QR
            # cv2.putText(image, f"QR: {data}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            pass

        # Invertir el ángulo si es necesario (según tus necesidades)
        angle2 = 360 - angle

        # Guardar los resultados con la fila y columna
        cell_index = row * cols + col  # Índice de la celda

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

        # Mostrar información en la imagen
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

        # Dibujar líneas punteadas dentro de la celda
        image = draw_dotted_line_in_cell(image, cell_center_x, cell_center_y, cell_width, cell_height)

    return detected_qrs, image

def draw_dotted_line_in_cell(image, cell_center_x, cell_center_y, cell_width, cell_height):
    """Dibuja una línea punteada roja dentro de la celda en los ejes del centro de la celda."""
    # Definir los límites de la celda
    cell_left = int(cell_center_x - cell_width // 2)
    cell_right = int(cell_center_x + cell_width // 2)
    cell_top = int(cell_center_y - cell_height // 2)
    cell_bottom = int(cell_center_y + cell_height // 2)

    # Dibujar línea punteada roja en el eje horizontal
    for x in range(cell_left, cell_right, 10):  # Incremento para punteado
        cv2.line(image, (x, cell_center_y), (x + 5, cell_center_y), (0, 0, 255), 1)

    # Dibujar línea punteada roja en el eje vertical
    for y in range(cell_top, cell_bottom, 10):  # Incremento para punteado
        cv2.line(image, (cell_center_x, y), (cell_center_x, y + 5), (0, 0, 255), 1)
    return image

def fill_cells(frame, matrix, alpha=0.7):
    """Rellena de color rojo translúcido los cuadrantes correspondientes a los valores '1' en la matriz."""
    rows, cols = len(matrix), len(matrix[0])
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    overlay = frame.copy()  # Hacemos una copia para aplicar el color translúcido

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] == 1:
                # Coordenadas del cuadrante
                x1, y1 = j * cell_width, i * cell_height
                x2, y2 = x1 + cell_width, y1 + cell_height
                # Rellenar el cuadrante con color rojo (translúcido)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

    # Aplicar transparencia a los rectángulos rojos
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

def highlight_start_end(frame, rows, cols):
    """Colorea en translúcido verde (0,0) y rojo (rows-1, cols-1)."""
    height, width, _ = frame.shape
    cell_height = height // rows
    cell_width = width // cols

    # Coordenadas del inicio (0, 0)
    x1_start, y1_start = 0, 0
    x2_start, y2_start = cell_width, cell_height
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1_start, y1_start), (x2_start, y2_start), (0, 255, 0), -1)  # Verde

    # Coordenadas del final (rows-1, cols-1)
    x1_end, y1_end = (cols - 1) * cell_width, (rows - 1) * cell_height
    x2_end, y2_end = x1_end + cell_width, y1_end + cell_height
    cv2.rectangle(overlay, (x1_end, y1_end), (x2_end, y2_end), (255, 0, 0), -1)  # Rojo

    # Agregar transparencia
    alpha = 0.5  # Nivel de transparencia
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

def on_trackbar_change(x):
    """Callback para manejar los cambios en las trackbars."""
    pass

# ==============================
# Función para Manejar la Entrada de Comandos
# ==============================

def command_input_thread():
    """Hilo para manejar la entrada de comandos desde la consola."""
    global running
    try:
        while running:
            print("\nControl del mBot:")
            print("w - Adelante")
            print("s - Atrás")
            print("a - Izquierda")
            print("d - Derecha")
            print("x - Detener")
            print("q - Salir")
            command = input("Ingresa un comando: ").strip().lower()

            if command == 'q':  # Salir del programa
                print("Cerrando conexión y terminando programa...")
                running = False
                comunicacionArduino.send_command(command)  # Enviar comando de cierre al Arduino
                break
            elif command in ['w', 's', 'a', 'd', 'x']:
                print(f"Enviando comando: {command}")  # Depuración
                comunicacionArduino.send_command(command)  # Enviar el comando válido
            else:
                print("Comando no reconocido")
    except KeyboardInterrupt:
        print("\nInterrupción por teclado.")
        running = False

# ==============================
# Hilo para Leer y Mostrar Respuestas
# ==============================

def serial_response_display_thread():
    """Hilo para leer y mostrar respuestas del Arduino desde la cola."""
    while running:
        try:
            response = comunicacionArduino.response_queue.get(timeout=0.1)
            print(f"Arduino dice: {response}")
        except queue.Empty:
            continue

def move_forward():
    """Envia el comando para mover el robot hacia adelante"""
    print("Moviendo hacia adelante...")
    comunicacionArduino.send_command('w')  # Comando para mover hacia adelante

# ==============================
# Inicio del Programa Principal
# ==============================

if __name__ == "__main__":
    # Iniciar el hilo de entrada de comandos
    input_thread = threading.Thread(target=command_input_thread, daemon=True)
    input_thread.start()

    # Iniciar el hilo de lectura de respuestas
    response_display_thread = threading.Thread(target=serial_response_display_thread, daemon=True)
    response_display_thread.start()

    # Abrir la fuente de video según la configuración
    if SOURCE_URL:
        cap = cv2.VideoCapture(URL)
        print(f"Usando la URL de DroidCam: {URL}")
        #move_forward()  # Mueve el robot hacia adelante
        #time.sleep(1)
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        print(f"Usando la cámara incorporada del computador (Índice: {CAMERA_INDEX})")
        #move_forward()
        #time.sleep(1)

    if not cap.isOpened():
        print("No se pudo conectar a la cámara seleccionada.")
        running = False
    else:
        print(f"Conexión exitosa. Analizando video con cuadrícula de {rows}x{cols}...")
        maze = maze_generate(rows, cols)
        print("Mapa generado:")
        for fila in maze:
            print(fila)

        # Crear ventana y trackbars
        cv2.namedWindow('Ajustes')
        cv2.createTrackbar('Canny Th1', 'Ajustes', canny_threshold1, 255, on_trackbar_change)
        cv2.createTrackbar('Canny Th2', 'Ajustes', canny_threshold2, 255, on_trackbar_change)
        cv2.createTrackbar('Dilatacion', 'Ajustes', 2, 15, on_trackbar_change)

        qr_detector = cv2.QRCodeDetector()

        while running:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar el video.")
                break

            # Obtener valores de las trackbars
            threshold1 = cv2.getTrackbarPos('Canny Th1', 'Ajustes')
            threshold2 = cv2.getTrackbarPos('Canny Th2', 'Ajustes')
            dilatacion = cv2.getTrackbarPos('Dilatacion', 'Ajustes')

            # Analizar el frame con los umbrales ajustados
            detected_qrs, frame_with_shapes = detect_qr_in_image(frame, rows, cols, qr_detector)

            # Dibujar la cuadrícula en el frame
            frame_with_grid = draw_grid(frame_with_shapes, rows, cols, thickness)

            # Rellenar las celdas del laberinto
            frame_filled = fill_cells(frame_with_grid, maze)

            # Resaltar las celdas de inicio y fin
            frame_highlighted = highlight_start_end(frame_filled, rows, cols)

            # Mostrar el frame con los ajustes
            cv2.imshow('Cuadrícula con análisis', frame_highlighted)

            # Mostrar información de los QR detectados
            for qr in detected_qrs:
                print(qr)

            

            # Presiona 'q' en la ventana de video para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Presionaste 'q'. Cerrando conexión y terminando programa...")
                running = False
                comunicacionArduino.send_command('q')  # Enviar comando de cierre al Arduino
                break

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()

    # Cerrar la conexión serial si está abierta
    comunicacionArduino.close_connection()
