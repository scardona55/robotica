import cv2
import numpy as np
import comunicacionArduino  # Asegúrate de que el nombre del archivo sea correcto
import math
import time

# ==============================
# Configuración de la Fuente de Video
# ==============================
SOURCE_URL = True  # Cambiar a True para DroidCam, False para cámara local
URL = "http://10.144.88.177:4747/video"
CAMERA_INDEX = 0  # Índice de la cámara local

# ==============================
# Parámetros de la Cuadrícula y Canny
# ==============================
rows = 7  # Número de filas
cols = 7  # Número de columnas

# ==============================
# Funciones Auxiliares
# ==============================

def detect_qr():
    """
    Detecta un código QR en la imagen y devuelve las coordenadas de sus esquinas.
    Utiliza OpenCV para detectar el QR y sus vértices.
    """
    # Selecciona la fuente de video (DroidCam o cámara local)
    if SOURCE_URL:
        cap = cv2.VideoCapture(URL)  # Usar URL de DroidCam
    else:
        cap = cv2.VideoCapture(CAMERA_INDEX)  # Usar cámara local

    if not cap.isOpened():
        print("Error: No se pudo acceder a la cámara.")
        return None

    # Captura de la imagen desde la fuente de video
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar imagen.")
        cap.release()
        return None

    # Crear el objeto detector de QR
    qr_decoder = cv2.QRCodeDetector()

    # Detectar el QR en la imagen
    data, points, _ = qr_decoder.detectAndDecode(frame)

    # Verifica si se detectó un QR
    if points is not None:
        # Si se detecta el QR, las coordenadas de las esquinas del QR están en 'points'
        points = points[0]  # Tomar las 4 esquinas del QR
        print("QR detectado en las coordenadas:", points)
        cap.release()  # Liberar la cámara
        return points
    else:
        print("No se detectó ningún código QR.")
        cap.release()  # Liberar la cámara
        return None

def calculate_angle(points):
    """
    Calcula el ángulo de inclinación en grados de un código QR dado.
    Se basa en las coordenadas de las esquinas superior izquierda y derecha.
    """
    # Extraer las coordenadas de las esquinas
    top_left = points[0]
    top_right = points[1]

    # Calcular el ángulo
    delta_y = top_right[1] - top_left[1]
    delta_x = top_right[0] - top_left[0]
    angle = np.arctan2(delta_y, delta_x)  # Ángulo en radianes

    # Convertir a grados
    angle_deg = np.degrees(angle)
    return angle_deg

def normalize_angle(angle):
    """
    Normaliza el ángulo al rango [0, 360).
    """
    angle = angle % 360
    if angle < 0:
        angle += 360
    return angle

def orient_qr_up():
    """
    Asegura que el QR esté orientado hacia arriba.
    Si el QR está inclinado, gira el robot hasta que quede en posición.
    """
    points = detect_qr()
    if points is not None:
        angle = calculate_angle(points)
        normalized_angle = normalize_angle(angle)

        print(f"Ángulo detectado: {normalized_angle} grados")

        # Si el ángulo no es 0° (es decir, no está orientado hacia arriba)
        while abs(normalized_angle) > 10:  # El umbral es 10 grados de tolerancia
            print("Girando el robot para alinear el QR...")
            comunicacionArduino.send_command("d")  # Ejecutar el comando de giro a la derecha
            time.sleep(1)  # Esperar para que el robot gire
            points = detect_qr()  # Volver a detectar el QR
            if points is not None:
                angle = calculate_angle(points)
                normalized_angle = normalize_angle(angle)
        print("QR orientado hacia arriba.")
    else:
        print("No se pudo detectar el QR, abortando ajuste de orientación.")
