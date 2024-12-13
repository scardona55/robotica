import cv2
import numpy as np
import random
import time
from comunicacionArduino import send_command_external
import threading


# URL de DroidCam
url = "http://192.168.1.5:4747/video"

# Parámetros de la cuadrícula
rows = 7  # Número de filas
cols = 7  # Número de columnas
thickness = 1  # Grosor de las líneas
shape_detected_this_session = False

# Valores iniciales de Canny
canny_threshold1 = 50
canny_threshold2 = 150

command_count = 0
last_detected_shape = None

def initialize_camera(url):
    """Función para inicializar la cámara con manejo de errores."""
    cap = cv2.VideoCapture(url)
    intentos = 0
    while not cap.isOpened() and intentos < 5:
        print(f"Intento {intentos + 1}: Conectando a la cámara...")
        cap = cv2.VideoCapture(url)
        intentos += 1
        time.sleep(1)  # Esperar un segundo entre intentos
    
    if not cap.isOpened():
        print("No se pudo conectar a la cámara después de varios intentos.")
        return None
    
    print(f"Conexión exitosa a la cámara. Resolución: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    return cap

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
    """
    Detecta círculos y triángulos en la imagen completa y calcula las celdas correspondientes.
    Limita la detección a una sola forma.
    """
    global last_detected_shape

    # Si ya se detectó una forma, no hacer más detecciones
    if last_detected_shape is not None:
        return [], image

    detected_shapes = []
    height, width, _ = image.shape
    cell_height = height // rows
    cell_width = width // cols

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detección de círculos
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
        param1=50, param2=30, minRadius=10, maxRadius=50
    )

    # Procesar círculos detectados
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for circle in circles:
            center_x, center_y, radius = circle
            row = center_y // cell_height
            col = center_x // cell_width
            cell_index = row * cols + col

            detected_shapes.append({
                "shape": "circle",
                "row": row,
                "col": col,
                "cell_index": cell_index,
                "x": center_x,
                "y": center_y
            })
            
            last_detected_shape = detected_shapes[0]
            print(f"Círculo detectado en celda: {cell_index}, Posición: ({row}, {col})")
            return detected_shapes, image

    # Detección de triángulos
    bordes = cv2.Canny(gray, threshold1, threshold2)
    kernel = np.ones((dilatacion, dilatacion), np.uint8)
    bordes = cv2.dilate(bordes, kernel)
    cv2.imshow("Bordes Modificado", bordes)
    figuras, jerarquia = cv2.findContours(bordes, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if jerarquia is not None:
        jerarquia = jerarquia[0]
        for i, contour in enumerate(figuras):
            if jerarquia[i][3] == -1:
                approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
                area = cv2.contourArea(contour)
                if len(approx) == 3 and area >= 500 and area < 3000:  # Triángulo
                    x, y, w, h = cv2.boundingRect(contour)
                    center_x, center_y = x + w // 2, y + h // 2
                    row = center_y // cell_height
                    col = center_x // cell_width
                    cell_index = row * cols + col

                    detected_shapes.append({
                        "shape": "triangle",
                        "row": row,
                        "col": col,
                        "cell_index": cell_index,
                        "x": center_x,
                        "y": center_y
                    })
                    
                    last_detected_shape = detected_shapes[0]
                    print(f"Triángulo detectado en celda: {cell_index}, Posición: ({row}, {col})")
                    return detected_shapes, image

    return [], image

# Añadir una función para reiniciar la detección cuando sea necesario
def reset_shape_detection():
    global shape_detected_this_session
    shape_detected_this_session = False

def fill_cells(frame, matrix, alpha=0.7):
    """Rellena de color negro translúcido los cuadrantes correspondientes a los valores '1' en la matriz."""
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
                # Rellenar el cuadrante con color negro (translúcido)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)

    # Aplicar transparencia a los rectángulos negros
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

def menu_control():
    """
    Muestra un menú por consola para controlar el robot y envía comandos.
    """
    global command_count, last_detected_shape
    
    print("\n--- Menú de Control ---")
    print("w - Adelante")
    print("s - Atrás")
    print("a - Izquierda")
    print("d - Derecha")
    print("x - Detener")
    print("q - Salir")
    print("-----------------------")
    opcion = input("Selecciona una opción: ").strip()

    # Mapear la opción a un comando
    comandos = {
        "w": "arriba",
        "s": "abajo",
        "a": "izquierda",
        "d": "derecha",
        "x": "detener",
    }

    if opcion == "q":
        print("Saliendo del control...")
        return False
    elif opcion in comandos:
        command = comandos[opcion]
        send_command_and_track(command)
        return True
    else:
        print("Opción inválida. Inténtalo de nuevo.")
        return True

def send_command_and_track(command):
    """
    Envía un comando y lleva un conteo de los comandos enviados.
    Imprime información de detección cada 3 comandos.
    """
    global command_count, last_detected_shape

    # Enviar comando a través de comunicacionArduino
    send_command_external(command)

    # Incrementar contador de comandos
    command_count += 1

    # Reiniciar la detección de formas cada cierto número de comandos
    if command_count % 3 == 0:
        last_detected_shape = None

def on_trackbar_change(x):
    """Callback para manejar los cambios en las trackbars."""
    pass

def handle_console_input():
    """Hilo para manejar la entrada de comandos desde la consola."""
    while True:
        command = input("Introduce un comando ('m' para menú, 'q' para salir): ").strip()
        if command == 'q':
            print("Saliendo...")
            break
        elif command == 'm':
            menu_control()  # Mostrar el menú solo si se solicita explícitamente
        else:
            print(f"Comando no reconocido: {command}")
    global stop_main_loop
    stop_main_loop = True  # Señal para detener la cámara y el bucle principal

def main_loop():
    global cap, command_count, last_detected_shape, stop_main_loop

    # Crear ventana y trackbars
    cv2.namedWindow('Ajustes')
    cv2.createTrackbar('Canny Th1', 'Ajustes', canny_threshold1, 255, on_trackbar_change)
    cv2.createTrackbar('Canny Th2', 'Ajustes', canny_threshold2, 255, on_trackbar_change)
    cv2.createTrackbar('Dilatacion', 'Ajustes', 2, 15, on_trackbar_change)
    
    maze = maze_generate(rows, cols)
    print("Laberinto generado:")
    for fila in maze:
        print(fila)

    while not stop_main_loop:
        # Verificar si la cámara está abierta
        if not cap or not cap.isOpened():
            print("Reconectando cámara...")
            cap = initialize_camera(url)
            if not cap:
                time.sleep(2)
                continue

        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame. Reintentando...")
            time.sleep(1)
            continue

        # Obtener valores de las trackbars
        threshold1 = cv2.getTrackbarPos('Canny Th1', 'Ajustes')
        threshold2 = cv2.getTrackbarPos('Canny Th2', 'Ajustes')
        dilatacion = cv2.getTrackbarPos('Dilatacion', 'Ajustes')

        # Analizar el frame con los umbrales ajustados
        detected_shapes, frame_with_shapes = detect_shapes_in_image(frame, rows, cols, threshold1, threshold2, dilatacion)
        
        # Dibujar la cuadrícula en el frame
        frame_with_grid = draw_grid(frame_with_shapes, rows, cols, thickness)
        frame = fill_cells(frame_with_grid, maze)
        frame = highlight_start_end(frame, rows, cols)

        # Mostrar el frame con los ajustes
        cv2.imshow('Cuadrícula con análisis', frame)

        # Manejo de la detección cada 3 comandos
        if command_count % 3 == 0 and command_count > 0:
            print(f"\n--- Detección de forma tras {command_count} comandos ---")
            print(f"Última forma detectada: {last_detected_shape if last_detected_shape else 'No detectada'}")
            print("-------------------------------------------\n")

        # Presiona 'q' para salir del video
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

# Inicialización global
stop_main_loop = False  # Variable para detener el bucle principal
try:
    cap = initialize_camera(url)
    if cap:
        # Crear un hilo para manejar la entrada de consola
        console_thread = threading.Thread(target=handle_console_input, daemon=True)
        console_thread.start()

        # Ejecutar el bucle principal
        main_loop()

        # Esperar a que el hilo de consola termine
        console_thread.join()
except Exception as e:
    print(f"Se produjo un error: {e}")
finally:
    # Liberar recursos y cerrar ventanas
    if cap and cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Recursos liberados. Aplicación finalizada correctamente.")
