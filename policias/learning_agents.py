# learning_agents.py

import random
import json
import logging
from collections import defaultdict
import time
import pickle
import os
import math
import comunicacionBluetooth  # Asegúrate de que este módulo esté correctamente implementado
from gridworld_utils import obtener_mapa_descriptivo, obtener_salida, obtener_obstaculos

logger = logging.getLogger(__name__)

class QLearningAgent:
    def __init__(self, rows, cols, role, alpha=0.1, gamma=0.9, epsilon=1.0, min_epsilon=0.1, decay_rate=0.995):
        """
        Inicializa el agente de Q-learning.

        :param rows: Número de filas del mapa.
        :param cols: Número de columnas del mapa.
        :param role: Rol del agente (0 = Policía, 1 = Ladrón).
        :param alpha: Tasa de aprendizaje.
        :param gamma: Factor de descuento.
        :param epsilon: Tasa de exploración inicial.
        :param min_epsilon: Tasa mínima de exploración.
        :param decay_rate: Tasa de decaimiento de epsilon.
        """
        self.rows = rows
        self.cols = cols
        self.role = role
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay_rate = decay_rate
        self.actions = ['up', 'down', 'left', 'right']
        self.Q = {}  # Tabla Q: {(fila, col): {accion: valor}}

    def state_to_index(self, row, col):
        return (row, col)

    def choose_action(self, state):
        """
        Elige una acción basada en la política epsilon-greedy.
        """
        if random.uniform(0, 1) < self.epsilon:
            accion = random.choice(self.actions)
            logger.debug(f"Exploración: seleccionada acción aleatoria '{accion}' en estado {state}")
        else:
            acciones_Q = self.Q.get(state, {})
            if not acciones_Q:
                accion = random.choice(self.actions)
                logger.debug(f"Explotación sin Q-values: seleccionada acción aleatoria '{accion}' en estado {state}")
            else:
                max_valor = max(acciones_Q.values())
                acciones_max = [accion for accion, valor in acciones_Q.items() if valor == max_valor]
                accion = random.choice(acciones_max)
                logger.debug(f"Explotación: seleccionada acción '{accion}' en estado {state} con Q-value {max_valor}")
        return accion

    def update_Q(self, state, accion, recompensa, siguiente_estado):
        """
        Actualiza el valor Q según la ecuación de Q-learning.
        """
        if state not in self.Q:
            self.Q[state] = {a: 0.0 for a in self.actions}
        if siguiente_estado not in self.Q:
            self.Q[siguiente_estado] = {a: 0.0 for a in self.actions}
        old_value = self.Q[state][accion]
        max_future = max(self.Q[siguiente_estado].values())
        self.Q[state][accion] += self.alpha * (recompensa + self.gamma * max_future - self.Q[state][accion])
        logger.debug(f"Actualizando Q[{state}][{accion}]: {old_value} -> {self.Q[state][accion]}")

    def get_reward(self, estado_actual, accion, maze, salida, police_position=None):
        """
        Define la función de recompensa basada en el rol del agente.

        :param estado_actual: Tupla (fila, columna) del estado actual.
        :param accion: Acción ejecutada.
        :param maze: Matriz del laberinto.
        :param salida: Tupla (fila, columna) del objetivo.
        :param police_position: Tupla (fila, columna) de la posición del policía (solo para el ladrón).
        :return: Recompensa numérica.
        """
        fila, columna = estado_actual
        if accion == 'up':
            fila_siguiente, columna_siguiente = fila - 1, columna
        elif accion == 'down':
            fila_siguiente, columna_siguiente = fila + 1, columna
        elif accion == 'left':
            fila_siguiente, columna_siguiente = fila, columna - 1
        elif accion == 'right':
            fila_siguiente, columna_siguiente = fila, columna + 1
        else:
            fila_siguiente, columna_siguiente = fila, columna  # Acción inválida

        # Recompensa por obstáculo o fuera de límites
        if (fila_siguiente < 0 or fila_siguiente >= self.rows or 
            columna_siguiente < 0 or columna_siguiente >= self.cols):
            return -100  # Penalización por moverse fuera del mapa
        if maze[fila_siguiente][columna_siguiente] == 1:
            return -100  # Penalización por chocar con un obstáculo

        # Recompensa por alcanzar el objetivo
        if (fila_siguiente, columna_siguiente) == salida:
            return 100

        # Penalización por cada paso
        recompensa = -1

        # Penalización adicional para el ladrón si el policía está cerca
        if self.role == 1 and police_position:
            distancia = math.hypot(police_position[0] - fila_siguiente, police_position[1] - columna_siguiente)
            if distancia <= 1.5:
                return -100

        return recompensa

    def train(self, maze, salida, police_position=None, episodes=1000, max_steps=100):
        """
        Entrena el agente usando Q-learning.

        :param maze: Matriz del laberinto.
        :param salida: Tupla (fila, columna) del objetivo.
        :param police_position: Tupla (fila, columna) de la posición del policía (solo para el ladrón).
        :param episodes: Número de episodios de entrenamiento.
        :param max_steps: Número máximo de pasos por episodio.
        """
        logger.info(f"Iniciando entrenamiento para {'Policía' if self.role == 0 else 'Ladrón'}")
        for episodio in range(1, episodes + 1):
            estado_actual = (0, 0)  # Estado inicial
            pasos = 0
            terminado = False

            for paso in range(max_steps):
                accion = self.choose_action(estado_actual)
                # Simular la acción para obtener el siguiente estado
                fila, columna = estado_actual
                if accion == 'up':
                    fila_siguiente, columna_siguiente = fila - 1, columna
                elif accion == 'down':
                    fila_siguiente, columna_siguiente = fila + 1, columna
                elif accion == 'left':
                    fila_siguiente, columna_siguiente = fila, columna - 1
                elif accion == 'right':
                    fila_siguiente, columna_siguiente = fila, columna + 1
                else:
                    fila_siguiente, columna_siguiente = fila, columna  # Acción inválida

                siguiente_estado = (fila_siguiente, columna_siguiente)
                recompensa = self.get_reward(estado_actual, accion, maze, salida, police_position)
                self.update_Q(estado_actual, accion, recompensa, siguiente_estado)

                # Actualizar estado
                estado_actual = siguiente_estado

                # Verificar si se ha alcanzado el objetivo
                if estado_actual == salida:
                    terminado = True
                    break

                pasos += 1

            # Decaimiento de epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.decay_rate

            # Logging cada 100 episodios
            if episodio % 100 == 0:
                logger.info(f"Episodio {episodio}/{episodes} completado.")

        logger.info(f"Entrenamiento completado para {'Policía' if self.role == 0 else 'Ladrón'}.")

    def save_policy(self, archivo_q='Q_table.pkl', archivo_politica='policy.json'):
        """
        Guarda la tabla Q y la política en archivos.
        """
        # Guardar la tabla Q
        with open(archivo_q, 'wb') as f:
            pickle.dump(self.Q, f)
        logger.info(f"Tabla Q guardada en {archivo_q}.")

        # Generar y guardar la política
        politica = self.generar_politica()
        with open(archivo_politica, 'w') as f:
            json.dump({str(k): v for k, v in politica.items()}, f, indent=4)
        logger.info(f"Política guardada en {archivo_politica}.")

    def cargar_policy(self, archivo_q='Q_table.pkl', archivo_politica='policy.json'):
        """
        Carga la tabla Q y la política desde archivos.
        """
        # Cargar la tabla Q
        if os.path.exists(archivo_q):
            with open(archivo_q, 'rb') as f:
                self.Q = pickle.load(f)
            logger.info(f"Tabla Q cargada desde {archivo_q}.")
        else:
            logger.warning(f"Archivo {archivo_q} no encontrado. Inicializando tabla Q vacía.")

        # Cargar la política
        if os.path.exists(archivo_politica):
            with open(archivo_politica, 'r') as f:
                politica = json.load(f)
            # Convertir claves de string a tuplas
            self.politica = {tuple(map(int, k.strip('()').split(', '))): v for k, v in politica.items()}
            logger.info(f"Política cargada desde {archivo_politica}.")
        else:
            logger.warning(f"Archivo {archivo_politica} no encontrado. Política no cargada.")
            self.politica = {}

    def generar_politica(self):
        """
        Genera una política basada en la tabla Q.
        """
        politica = {}
        for estado, acciones in self.Q.items():
            mejor_accion = max(acciones, key=acciones.get)
            politica[estado] = mejor_accion
        return politica

    def get_action_from_policy(self, estado):
        """
        Obtiene la mejor acción basada en la política cargada.
        """
        return self.politica.get(estado, random.choice(self.actions))

def inicializar_Q(mapa):
    """
    Inicializa la tabla Q para todos los estados y acciones posibles.

    Parámetros:
    mapa (list of list of int): Mapa original del laberinto (0: pasillo, 1: obstáculo).

    Retorna:
    dict: Tabla Q inicializada con 0.0 para todas las acciones en cada estado.
    """
    Q = {}
    rows = len(mapa)
    cols = len(mapa[0])
    acciones = ['up', 'down', 'left', 'right']
    for i in range(rows):
        for j in range(cols):
            if mapa[i][j] != 1:  # No hay acciones desde obstáculos
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
    Actualiza el valor Q según la ecuación de Q-learning.

    :param Q: Tabla Q.
    :param estado: Estado actual.
    :param accion: Acción ejecutada.
    :param recompensa: Recompensa obtenida.
    :param siguiente_estado: Estado siguiente.
    :param alpha: Tasa de aprendizaje.
    :param gamma: Factor de descuento.
    """
    if siguiente_estado not in Q:
        Q[siguiente_estado] = {a: 0.0 for a in ['up', 'down', 'left', 'right']}
    max_Q_siguiente = max(Q[siguiente_estado].values())
    Q[estado][accion] += alpha * (recompensa + gamma * max_Q_siguiente - Q[estado][accion])

def obtener_recompensa(estado, accion, mapa, salida):
    """
    Define la función de recompensa basada en el rol del agente.

    :param estado: Estado actual (fila, columna).
    :param accion: Acción ejecutada.
    :param mapa: Mapa original del laberinto (0: pasillo, 1: obstáculo).
    :param salida: Posición de la salida (fila, columna).
    :return: Recompensa numérica.
    """
    fila, columna = estado
    # Simular la acción para ver el siguiente estado
    if accion == 'up':
        fila_siguiente, columna_siguiente = fila - 1, columna
    elif accion == 'down':
        fila_siguiente, columna_siguiente = fila + 1, columna
    elif accion == 'left':
        fila_siguiente, columna_siguiente = fila, columna - 1
    elif accion == 'right':
        fila_siguiente, columna_siguiente = fila, columna + 1
    else:
        fila_siguiente, columna_siguiente = fila, columna  # Acción inválida

    # Verificar si está fuera del mapa
    if fila_siguiente < 0 or fila_siguiente >= len(mapa) or columna_siguiente < 0 or columna_siguiente >= len(mapa[0]):
        return -100  # Penalización por intentar moverse fuera del mapa

    # Verificar si es un obstáculo
    if mapa[fila_siguiente][columna_siguiente] == 1:
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
        fila_siguiente, columna_siguiente = fila - 1, columna
    elif accion == 'down':
        fila_siguiente, columna_siguiente = fila + 1, columna
    elif accion == 'left':
        fila_siguiente, columna_siguiente = fila, columna - 1
    elif accion == 'right':
        fila_siguiente, columna_siguiente = fila, columna + 1
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
    for estado, acciones in Q.items():
        mejor_accion = max(acciones, key=acciones.get)
        politica[estado] = mejor_accion
    return politica

def guardar_Q_politica(Q, politica, archivo_q='Q_table.pkl', archivo_politica='policy.json'):
    """
    Guarda la tabla Q y la política en archivos.
    """
    # Guardar la tabla Q
    with open(archivo_q, 'wb') as f:
        pickle.dump(Q, f)
    logger.info(f"Tabla Q guardada en {archivo_q}.")

    # Guardar la política
    with open(archivo_politica, 'w') as f:
        json.dump({str(k): v for k, v in politica.items()}, f, indent=4)
    logger.info(f"Política guardada en {archivo_politica}.")

def cargar_Q_politica(archivo_q='Q_table.pkl', archivo_politica='policy.json'):
    """
    Carga la tabla Q y la política desde archivos.
    """
    # Cargar la tabla Q
    if os.path.exists(archivo_q):
        with open(archivo_q, 'rb') as f:
            Q = pickle.load(f)
        logger.info(f"Tabla Q cargada desde {archivo_q}.")
    else:
        logger.warning(f"Archivo {archivo_q} no encontrado. Inicializando tabla Q vacía.")
        Q = {}

    # Cargar la política
    if os.path.exists(archivo_politica):
        with open(archivo_politica, 'r') as f:
            politica = json.load(f)
        # Convertir claves de string a tuplas
        politica = {tuple(map(int, k.strip('()').split(', '))): v for k, v in politica.items()}
        logger.info(f"Política cargada desde {archivo_politica}.")
    else:
        logger.warning(f"Archivo {archivo_politica} no encontrado. Política no cargada.")
        politica = {}

    return Q, politica

def entrenar_Q_learning(maze, salida, role=0, police_position=None, alpha=0.1, gamma=0.9, 
                        epsilon=1.0, min_epsilon=0.1, decay_rate=0.995, episodes=1000):
    """
    Entrena un agente de Q-learning.

    :param maze: Mapa del laberinto.
    :param salida: Posición de la salida.
    :param role: Rol del agente (0 = Policía, 1 = Ladrón).
    :param police_position: Posición del policía (solo para el ladrón).
    :param alpha: Tasa de aprendizaje.
    :param gamma: Factor de descuento.
    :param epsilon: Tasa de exploración inicial.
    :param min_epsilon: Tasa mínima de exploración.
    :param decay_rate: Tasa de decaimiento de epsilon.
    :param episodes: Número de episodios de entrenamiento.
    :return: Agente entrenado.
    """
    agent = QLearningAgent(len(maze), len(maze[0]), role, alpha, gamma, epsilon, min_epsilon, decay_rate)
    agent.train(maze, salida, police_position, episodes)
    agent.save_policy(
        archivo_q=f"Q_table_role_{role}.pkl",
        archivo_politica=f"policy_role_{role}.json"
    )
    return agent

def cargar_agente(role=0):
    """
    Carga un agente entrenado desde archivos.

    :param role: Rol del agente (0 = Policía, 1 = Ladrón).
    :return: Agente cargado.
    """
    agent = QLearningAgent(0, 0, role)  # Las dimensiones se actualizarán después de cargar la tabla Q
    agent.cargar_policy(
        archivo_q=f"Q_table_role_{role}.pkl",
        archivo_politica=f"policy_role_{role}.json"
    )
    return agent
