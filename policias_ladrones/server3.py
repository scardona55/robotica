from flask import Flask, jsonify, request, Response
import cv2
import numpy as np
import math
import threading

app = Flask(__name__)

# Inicializar variables globales para la cámara y la última imagen capturada
camera = cv2.VideoCapture("http://192.168.1.5:4747/video")  # Abrir la cámara
#camera = cv2.VideoCapture(0)
latest_frame = None
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
parameters = cv2.aruco.DetectorParameters()
rows = 5
cols = 5
maze=[[0,0,0,1,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,0,0,0,0]]
roles={
    8:0, #El valor 8 será el policía
    9:1  #El valor 1 será el ladrón
}
bandera=0

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

# Función para calcular el ángulo de inclinación
"""
def calculate_angle(corners):
    # Extraer las coordenadas de las esquinas del marcador
    top_left, top_right, bottom_right, bottom_left = corners.reshape((4, 2))

    # Calcular el vector desde la esquina superior izquierda (0° referencia)
    vector = top_right - top_left

    # Calcular el ángulo en radianes y convertir a grados
    angle = math.degrees(math.atan2(vector[1], vector[0]))

    # Convertir el ángulo a rango [0, 360]
    if angle < 0:
        angle += 360

    return angle
"""
# Función para calcular el ángulo de inclinación
# Función para calcular el ángulo de inclinación
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

# Dibujar flechas indicando orientación
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

# Función para capturar imágenes continuamente
def capture_frames():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if ret:
            latest_frame = frame

# Iniciar el hilo de captura de frames
capture_thread = threading.Thread(target=capture_frames, daemon=True)
capture_thread.start()

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


@app.route('/video_feed', methods=['GET'])
def video_feed():
    def generate():
        while True:
            if latest_frame is not None:
                # Copiar el frame para no modificar la referencia global
                frame_copy = latest_frame.copy()
                # Dibujar la grilla y detectar formas

                detect_shapes_in_image(frame_copy, rows, cols)
                draw_grid(frame_copy, rows, cols,maze)
                #highlight_start_end(frame_copy,rows,cols)
                # Codificar la imagen como JPEG
                _, buffer = cv2.imencode('.jpg', frame_copy)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(generate(), content_type='multipart/x-mixed-replace; boundary=frame')

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

            # Mostrar ID, posición y orientación en consola
            #print(f"ID: {marker_id}, Centro: ({center[0]:.2f}, {center[1]:.2f}), Ángulo: {angle:.2f}°")

            # Dibujar flechas indicando orientación
            draw_arrows(image, center, angle)

            # Mostrar ID y ángulo en la imagen
            cv2.putText(image, f"ID: {marker_id}", (int(center[0] - 50), int(center[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(image, f"{int(angle)}''", (int(center[0] - 50), int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


            row = int(math.floor(center[1]) // cell_height)
            col = int(math.floor(center[0]) // cell_width)
            cell_index = row * cols + col
            cell_center_x = int((col + 0.5) * cell_width)
            cell_center_y = int((row + 0.5) * cell_height)

            cv2.putText(image, f"{cell_index}", (int(center[0]-10), int(center[1]+20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)



            role=-1
            if marker_id==8:
                role=bandera
            elif marker_id==9:
                role = 1-bandera
            roleTxt="Policia" if role==0 else str("Ladron")
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
                "role":role
            })

    print(detected_shapes)
    return detected_shapes

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
        shapes = detect_shapes_in_image(frame_copy, rows, cols)
        shapes=validate_and_convert_dict(shapes)
        return jsonify(shapes)
    else:
        return jsonify({"error": "No frame available from the camera"}), 500


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

def validate_and_process_dict(data):
    """
    Valida y procesa un diccionario para garantizar que:
      - Todos los valores numéricos sean convertidos a enteros.
      - Todos los valores sean serializables a JSON.

    :param data: Lista de diccionarios a validar y procesar.
    :return: Lista de diccionarios procesados.
    """
    processed_data = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Cada elemento debe ser un diccionario.")

        processed_item = {}
        for key, value in item.items():
            # Convertir números a enteros
            if isinstance(value, (int, float)):
                processed_item[key] = int(value)
            # Verificar otros tipos serializables
            elif isinstance(value, (str, list, dict, bool)) or value is None:
                processed_item[key] = value
            else:
                raise TypeError(f"El valor de la clave '{key}' no es serializable: {value}")

        processed_data.append(processed_item)

    return processed_data


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
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000)
    finally:
        camera.release()  # Liberar la cámara al cerrar el servidor