import random
import json
import logging
from collections import defaultdict
import time
import comunicacionBluetooth
from gridworld_utils import obtener_mapa_descriptivo, obtener_salida, obtener_obstaculos

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

def seleccionar_accion(Q, estado, epsilon):
    """
    Selecciona una acción usando la política epsilon-greedy.
    """
    if random.uniform(0, 1) < epsilon:
        # Exploración: seleccionar una acción aleatoria válida
        accion = random.choice(list(Q[estado].keys()))
    else:
        # Explotación: seleccionar la acción con el mayor valor Q
        max_valor = max(Q[estado].values())
        acciones_max = [accion for accion, valor in Q[estado].items() if valor == max_valor]
        accion = random.choice(acciones_max)
    return accion

def actualizar_Q(Q, estado, accion, recompensa, siguiente_estado, alpha, gamma):
    """
    Q (dict): Tabla Q.
    estado (tuple): Estado actual (fila, columna).
    accion (str): Acción ejecutada.
    recompensa (float): Recompensa obtenida.
    siguiente_estado (tuple): Estado resultante después de la acción.
    alpha (float): Tasa de aprendizaje.
    gamma (float): Factor de descuento.
    """
    if siguiente_estado not in Q:
        # Si el siguiente estado no está en Q, lo inicializamos
        Q[siguiente_estado] = {a: 0.0 for a in ['up', 'down', 'left', 'right']}
    max_Q_siguiente = max(Q[siguiente_estado].values())
    Q[estado][accion] = Q[estado][accion] + alpha * (recompensa + gamma * max_Q_siguiente - Q[estado][accion])

def obtener_recompensa(estado, accion, mapa, salida):
    """
    estado (tuple): Estado actual (fila, columna).
    accion (str): Acción ejecutada.
    mapa (list of list of str): Mapa descriptivo.
    salida (tuple): Posición de la salida.
    """
    fila, columna = estado
    # Simular la acción para ver el siguiente estado
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
        fila_siguiente, columna_siguiente = fila, columna  # Acción inválida

    # Verificar si está fuera del mapa
    if fila_siguiente < 0 or fila_siguiente >= len(mapa) or columna_siguiente < 0 or columna_siguiente >= len(mapa[0]):
        return -100  # Penalización por intentar moverse fuera del mapa

    # Verificar si es un obstáculo
    if mapa[fila_siguiente][columna_siguiente] == 'O':
        return -100  # Penalización por chocar con un obstáculo

    # Verificar si ha llegado a la salida
    if (fila_siguiente, columna_siguiente) == salida:
        return 100  # Recompensa por llegar a la salida

    # Penalización por cada paso para incentivar caminos más cortos
    return -1

def obtener_siguiente_estado(estado, accion, maze):
    """
    Obtiene el siguiente estado basado en la acción ejecutada.
    Parámetros:
    estado (tuple): Estado actual (fila, columna).
    accion (str): Acción ejecutada ('up', 'down', 'left', 'right').
    maze (list of list of int): Mapa original del laberinto (0: pasillo, 1: obstáculo).
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
        fila_siguiente, columna_siguiente = fila, columna  # Acción inválida

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

def guardar_Q_politica(Q, politica, archivo_q='Q_table.json', archivo_politica='policy.json'):
    """
    Guarda la tabla Q y la política en archivos JSON.
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

def cargar_Q_politica(archivo_q='Q_table.json', archivo_politica='policy.json'):
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

def entrenar_Q_learning(maze, Q, salida, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.1, decay_rate=0.995, episodes=1000):

    logger.info("Iniciando entrenamiento de Q-learning...")
    mapa = obtener_mapa_descriptivo(maze)

    for episodio in range(episodes):
        estado_actual = (0, 0)  # Estado inicial
        pasos = 0
        max_pasos = len(maze) * len(maze[0])  # Limitar pasos por episodio
        terminado = False

        while not terminado and pasos < max_pasos:
            accion = seleccionar_accion(Q, estado_actual, epsilon)
            siguiente_estado = obtener_siguiente_estado(estado_actual, accion, maze)
            recompensa = obtener_recompensa(estado_actual, accion, mapa, salida)
            actualizar_Q(Q, estado_actual, accion, recompensa, siguiente_estado, alpha, gamma)
            estado_actual = siguiente_estado

            if estado_actual == salida:
                terminado = True
            pasos += 1

        # Reducir la tasa de exploración
        if epsilon > min_epsilon:
            epsilon *= decay_rate

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
    for _ in range(8):
        comunicacionBluetooth.send_command('w')
        time.sleep(0.05)  # Pausa de 50 ms entre comandos
    logger.info("Movimiento hacia adelante ejecutado.")

def move_back():
    for _ in range(8):
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
