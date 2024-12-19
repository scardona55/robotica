import random
import json
import logging
from collections import defaultdict
import time
import comunicacionBluetooth
from gridworld_utils import obtener_mapa_descriptivo, obtener_salida, obtener_obstaculos
from qlearn import obtener_recompensa, seleccionar_accion

logger = logging.getLogger(__name__)

def inicializar_Q(mapa):
    """
    Inicializa la tabla Q para todos los estados y acciones posibles.

    Parámetros:
    mapa (list of list of str): Mapa descriptivo con 'S', 'E', 'O', 'P'.

    Retorna:
    dict: Tabla Q inicializada con 0.0 para todas las acciones en cada estado.
    """
    Q = {}
    rows = len(mapa)
    cols = len(mapa[0])
    acciones = ['up', 'down', 'left', 'right']
    for i in range(rows):
        for j in range(cols):
            if mapa[i][j] != 'O':  # No hay acciones desde obstáculos
                estado = (i, j)
                Q[estado] = {accion: 0.0 for accion in acciones}
    return Q

def actualizar_Q(Q, estado, accion, recompensa, siguiente_estado, siguiente_accion, alpha, gamma):
    """
    Actualiza el valor Q para un estado y acción dada.

    Parámetros:
    Q (dict): Tabla Q.
    estado (tuple): Estado actual (fila, columna).
    accion (str): Acción ejecutada.
    recompensa (float): Recompensa obtenida.
    siguiente_estado (tuple): Estado resultante después de la acción.
    siguiente_accion (str): Acción que se tomará en el siguiente estado.
    alpha (float): Tasa de aprendizaje.
    gamma (float): Factor de descuento.

    Retorna:
    None
    """
    Q[estado][accion] = Q[estado][accion] + alpha * (recompensa + gamma * Q[siguiente_estado][siguiente_accion] - Q[estado][accion])

def obtener_siguiente_estado(estado, accion, maze):
    """
    Obtiene el siguiente estado basado en la acción ejecutada.

    Parámetros:
    estado (tuple): Estado actual (fila, columna).
    accion (str): Acción ejecutada ('up', 'down', 'left', 'right').
    maze (list of list of int): Mapa original del laberinto (0: pasillo, 1: obstáculo).

    Retorna:
    tuple: Nuevo estado (fila, columna).
    """
    fila, columna = estado
    if accion == 'up':
        fila_siguiente = fila - 1
        columna_siguiente = columna
    elif accion == 'down':
        fila_siguiente = fila + 1
        columna_siguiente = columna
    elif accion == 'left':
        fila_siguiente = fila
        columna_siguiente = columna - 1
    elif accion == 'right':
        fila_siguiente = fila
        columna_siguiente = columna + 1
    else:
        fila_siguiente, columna_siguiente = fila, columna  #La acción es inválida

    # Verificar límites y obstáculos
    if fila_siguiente < 0 or fila_siguiente >= len(maze) or columna_siguiente < 0 or columna_siguiente >= len(maze[0]):
        return estado  # No moverse si está fuera de los límites
    if maze[fila_siguiente][columna_siguiente] == 1:
        return estado  # No moverse si hay un obstáculo

    return (fila_siguiente, columna_siguiente)

def generar_politica(Q):
    """
    Genera una política basada en la tabla Q.

    Parámetros:
    Q (dict): Tabla Q.

    Retorna:
    dict: Política derivada de Q.
    """
    politica = {}
    for estado in Q:
        mejor_accion = max(Q[estado], key=Q[estado].get)
        politica[estado] = mejor_accion  # Usar tupla para las claves
    return politica

def guardar_Q_politica(Q, politica, archivo_q='Q_table_sarsa.json', archivo_politica='policy_sarsa.json'):
    """
    Guarda la tabla Q y la política en archivos JSON.

    Parámetros:
    Q (dict): Tabla Q.
    politica (dict): Política derivada de Q.
    archivo_q (str): Nombre del archivo para Q-table.
    archivo_politica (str): Nombre del archivo para la política.

    Retorna:
    None
    """
    # Convertir las claves de Q a strings para JSON
    Q_serializable = {str(k): v for k, v in Q.items()}
    with open(archivo_q, 'w') as f:
        json.dump(Q_serializable, f, indent=4)

    # Convertir las claves de la política a strings para JSON
    politica_serializable = {str(k): v for k, v in politica.items()}
    with open(archivo_politica, 'w') as f:
        json.dump(politica_serializable, f, indent=4)

    logger.info(f"Q-table y política guardadas en {archivo_q} y {archivo_politica} respectivamente.")

def cargar_Q_politica(archivo_q='Q_table_sarsa.json', archivo_politica='policy_sarsa.json'):
    """
    Carga la tabla Q y la política desde archivos JSON.

    Parámetros:
    archivo_q (str): Nombre del archivo para Q-table.
    archivo_politica (str): Nombre del archivo para la política.

    Retorna:
    tuple: (Q, politica)
    """
    with open(archivo_q, 'r') as f:
        Q = json.load(f)
        # Convertir las claves de string a tuplas
        Q = {tuple(map(int, k.strip('()').split(', '))): v for k, v in Q.items()}

    with open(archivo_politica, 'r') as f:
        politica = json.load(f)
        # Convertir las claves de string a tuplas
        politica = {tuple(map(int, k.strip('()').split(', '))): v for k, v in politica.items()}

    logger.info(f"Q-table y política cargadas desde {archivo_q} y {archivo_politica} respectivamente.")
    return Q, politica

def entrenar_SARSA(maze, Q, salida, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, episodes=1000):
    """
    Entrena la tabla Q utilizando el algoritmo SARSA.

    Parámetros:
    maze (list of list of int): Mapa original del laberinto.
    Q (dict): Tabla Q inicializada.
    salida (tuple): Posición de la salida.
    alpha (float): Tasa de aprendizaje.
    gamma (float): Factor de descuento.
    epsilon (float): Tasa de exploración inicial.
    epsilon_min (float): Tasa de exploración mínima.
    epsilon_decay (float): Tasa de decaimiento de epsilon.
    episodes (int): Número de episodios de entrenamiento.

    Retorna:
    tuple: (Q, politica)
    """
    logger.info("Iniciando entrenamiento de SARSA...")
    mapa = obtener_mapa_descriptivo(maze)

    for episodio in range(episodes):
        estado_actual = (0, 0)  # Estado inicial
        accion_actual = seleccionar_accion(Q, estado_actual, epsilon)
        pasos = 0
        max_pasos = len(maze) * len(maze[0])  # Limitar pasos por episodio
        terminado = False

        while not terminado and pasos < max_pasos:
            siguiente_estado = obtener_siguiente_estado(estado_actual, accion_actual, maze)
            recompensa = obtener_recompensa(estado_actual, accion_actual, mapa, salida)
            accion_siguiente = seleccionar_accion(Q, siguiente_estado, epsilon)

            # Actualizar Q-table usando SARSA
            actualizar_Q(Q, estado_actual, accion_actual, recompensa, siguiente_estado, accion_siguiente, alpha, gamma)

            estado_actual = siguiente_estado
            accion_actual = accion_siguiente

            if estado_actual == salida:
                terminado = True
            pasos += 1

        # Reducir la tasa de exploración
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Mostrar progreso cada 100 episodios
        if (episodio + 1) % 100 == 0:
            logger.info(f"Época {episodio + 1}/{episodes} completada.")

    # Generar política
    politica = generar_politica(Q)
    # Guardar Q-table y política
    guardar_Q_politica(Q, politica)
    logger.info("Entrenamiento completado y política generada.")

    return Q, politica

def move_forward():
    for _ in range(10):
        comunicacionBluetooth.send_command('w')
        time.sleep(0.05)  # Pausa de 50 ms entre comandos
    logger.info("Movimiento hacia adelante ejecutado.")

def move_back():
    for _ in range(10):
        comunicacionBluetooth.send_command('s')
        time.sleep(0.05)  # Pausa de 50 ms entre comandos
    logger.info("Movimiento hacia atrás ejecutado.")

def turn_left():
    for _ in range(40):
        comunicacionBluetooth.send_command('d')
        time.sleep(0.05)  # Pausa de 50 ms entre comandos
    logger.info("Giro a la izquierda ejecutado.")

def turn_right():
    for _ in range(40):
        comunicacionBluetooth.send_command('a')
        time.sleep(0.05)  # Pausa de 50 ms entre comandos
    logger.info("Giro a la derecha ejecutado.")
