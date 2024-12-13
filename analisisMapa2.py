# analisisMapa.py
import cv2
import numpy as np
import random
import threading
import comunicacionArduino  # Asegúrate de que el nombre del archivo sea correcto

# ==============================
# Configuración de la Fuente de Video
# ==============================

# Modo de fuente de video
# Establece SOURCE_URL en True para usar la URL de DroidCam
# Establece SOURCE_URL en False para usar la cámara incorporada del computador
SOURCE_URL = False  # Cambiar a True para DroidCam, False para cámara local

# URL de DroidCam
URL = "http://192.168.37.118:4747/video"

# Índice de la cámara incorporada (normalmente 0)
CAMERA_INDEX = 0

# ==============================
# Parámetros de la Cuadrícula y Canny
# ==============================

# Parámetros de la cuadrícula
rows = 7  # Número de filas
cols = 7  # Número de columnas
thickness = 1  # Grosor de las líneas

# Valores iniciales de Canny
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

def detect_shapes_in_image(image, rows, cols, threshold1, threshold2, dilatacion):
    """Detecta círculos y triángulos en la imagen completa y calcula las celdas correspondientes."""
    detected_shapes = []
    height, width, _ = image.shape
    cell_height = height // rows
    cell_width = width // cols

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Umbral inverso para detectar regiones negras
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # Detección de círculos con HoughCircles
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=10, maxRadius=50
    )

    # Procesar círculos detectados
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")  # Convertir a enteros
        for circle in circles:
            center_x, center_y, radius = circle
            row = center_y // cell_height
            col = center_x // cell_width
            cell_index = row * cols + col  # Índice de la celda

            # Dibujar círculo
            cv2.circle(image, (center_x, center_y), radius, (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{cell_index}",
                (center_x - 10, center_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            cv2.putText(
                image,
                f"{center_x},{center_y}",
                (center_x - 30, center_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
            detected_shapes.append({
                "shape": "circle",
                "row": row,
                "col": col,
                "cell_index": cell_index,
                "x": center_x,
                "y": center_y
            })

    bordes = cv2.Canny(gray, threshold1, threshold2)
    kernel = np.ones((dilatacion, dilatacion), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    cv2.imshow("Bordes Modificado", bordes)
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if jerarquia is not None:
        jerarquia = jerarquia[0]
    i = 0
    for contour in figuras:
        if jerarquia[i][3] == -1:
            approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
            area = cv2.contourArea(contour)
            if len(approx) == 3 and area >= 500 and area < 3000:  # Triángulo
                x, y, w, h = cv2.boundingRect(contour)
                center_x, center_y = x + w // 2, y + h // 2  # Centro aproximado del triángulo
                row = center_y // cell_height
                col = center_x // cell_width
                cell_index = row * cols + col  # Índice de la celda

                # Dibujar triángulo
                cv2.drawContours(image, [approx], -1, (255, 0, 0), 2)
                cv2.putText(
                    image,
                    f"{cell_index}",
                    (center_x, center_y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    image,
                    f"{center_x},{center_y}",
                    (center_x - 30, center_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
                detected_shapes.append({
                    "shape": "triangle",
                    "row": row,
                    "col": col,
                    "cell_index": cell_index,
                    "x": center_x,
                    "y": center_y
                })
                # Comentado para detectar múltiples triángulos
                # break
        i += 1

    return detected_shapes, image

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
                comunicacionArduino.send_command(command)  # Opcional: Notificar al Arduino
                break
            elif command in ['w', 's', 'a', 'd', 'x']:
                print(f"Enviando comando: {command}")  # Depuración
                comunicacionArduino.send_command(command)  # Enviar el comando válido
            else:
                print("Comando no reconocido")

            # No es necesario llamar a read_from_arduino() aquí, ya que el hilo de lectura lo maneja
    except KeyboardInterrupt:
        print("\nInterrupción por teclado.")
        running = False

# ==============================
# Inicio del Programa Principal
# ==============================

if __name__ == "__main__":
    # Iniciar el hilo de entrada de comandos
    input_thread = threading.Thread(target=command_input_thread, daemon=True)
    input_thread.start()

    # Abrir la fuente de video según la configuración
    if SOURCE_URL:
        cap = cv2.VideoCapture(URL)
        print(f"Usando la URL de DroidCam: {URL}")
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        print(f"Usando la cámara incorporada del computador (Índice: {CAMERA_INDEX})")

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
            detected_shapes, frame_with_shapes = detect_shapes_in_image(frame, rows, cols, threshold1, threshold2, dilatacion)
            print(detected_shapes)
            # Dibujar la cuadrícula en el frame
            frame_with_grid = draw_grid(frame_with_shapes, rows, cols, thickness)

            frame_filled = fill_cells(frame_with_grid, maze)
            frame_highlighted = highlight_start_end(frame_filled, rows, cols)
            # Mostrar el frame con los ajustes
            cv2.imshow('Cuadrícula con análisis', frame_highlighted)

            # Presiona 'q' en la ventana de video para salir
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Presionaste 'q'. Cerrando conexión y terminando programa...")
                running = False
                comunicacionArduino.send_command('q')  # Opcional: Notificar al Arduino
                break

    # Libera recursos
    cap.release()
    cv2.destroyAllWindows()

    # Cerrar la conexión serial si está abierta
    comunicacionArduino.close_connection()
