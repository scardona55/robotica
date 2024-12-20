# learning_agents.py

import random
import json
import logging
from collections import defaultdict
import pickle
import os
import math

# Configuración del logger
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("qlearning.log"),
        logging.StreamHandler()
    ]
)
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
        self.politica = {}

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

    def simular_accion(self, fila, columna, accion):
        """
        Simula una acción y retorna el siguiente estado.
        """
        if accion == 'up':
            return fila - 1, columna
        elif accion == 'down':
            return fila + 1, columna
        elif accion == 'left':
            return fila, columna - 1
        elif accion == 'right':
            return fila, columna + 1
        else:
            logger.warning(f"Acción inválida: {accion}. Manteniendo posición.")
            return fila, columna  # Acción inválida

    def obtener_siguiente_estado(self, estado, accion, maze):
        """
        Obtiene el siguiente estado basado en la acción ejecutada, asegurando que esté dentro de los límites.
        """
        fila, columna = estado
        fila_siguiente, columna_siguiente = self.simular_accion(fila, columna, accion)

        if fila_siguiente < 0 or fila_siguiente >= self.rows or columna_siguiente < 0 or columna_siguiente >= self.cols:
            logger.debug(f"Movimiento fuera de límites: {fila_siguiente}, {columna_siguiente}. Manteniendo estado: {estado}")
            return estado  # No moverse si está fuera de los límites
        if maze[fila_siguiente][columna_siguiente] == 1:
            logger.debug(f"Chocando con obstáculo en: {fila_siguiente}, {columna_siguiente}. Manteniendo estado: {estado}")
            return estado  # No moverse si hay un obstáculo

        return (fila_siguiente, columna_siguiente)

    def get_reward(self, estado_actual, accion, maze, other_agent_position=None):
        """
        Define la función de recompensa basada en el rol del agente.
        
        :param estado_actual: Tupla (fila, columna) del estado actual.
        :param accion: Acción ejecutada.
        :param maze: Matriz del laberinto.
        :param other_agent_position: Tupla (fila, columna) de la posición del otro agente.
        :return: Recompensa numérica y flag de terminación.
        """
        fila, columna = estado_actual
        fila_siguiente, columna_siguiente = self.simular_accion(fila, columna, accion)

        # Recompensa por obstáculo o fuera de límites
        if (fila_siguiente < 0 or fila_siguiente >= self.rows or 
            columna_siguiente < 0 or columna_siguiente >= self.cols):
            return -100, False  # Penalización por moverse fuera del mapa
        if maze[fila_siguiente][columna_siguiente] == 1:
            return -100, False  # Penalización por chocar con un obstáculo

        # Definir el siguiente estado
        siguiente_estado = (fila_siguiente, columna_siguiente)

        # Penalización por cada paso
        recompensa = -1

        # Verificar captura
        if other_agent_position and siguiente_estado == other_agent_position:
            if self.role == 0:
                # Policía captura al ladrón
                return 100, True  # Recompensa positiva y terminar el episodio
            elif self.role == 1:
                # Ladrón es capturado
                return -100, True  # Penalización negativa y terminar el episodio

        return recompensa, False  # Recompensa por paso y no terminar

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

    def obtener_estado_inicial(self, maze):
        """
        Encuentra un estado inicial válido (no obstáculo).
        """
        valid_states = [(fila, columna) for fila in range(self.rows) for columna in range(self.cols) if maze[fila][columna] == 0]
        if not valid_states:
            raise ValueError("No hay estados iniciales válidos en el mapa.")
        estado_inicial = random.choice(valid_states)
        logger.debug(f"Estado inicial seleccionado: {estado_inicial}")
        return estado_inicial

    def obtener_posicion_other_agent(self, other_agent_positions, maze):
        """
        Determina la posición del otro agente en el paso actual.
        Aquí, se puede implementar una lógica para mover al otro agente.
        Por simplicidad, asumimos que el otro agente se mueve aleatoriamente evitando obstáculos y manteniéndose dentro del mapa.
        """
        if not other_agent_positions:
            # Inicializar otra posición aleatoria válida
            other_agent_position = self.obtener_estado_inicial(maze)
            other_agent_positions.append(other_agent_position)
        else:
            # Mover al otro agente
            current_position = other_agent_positions[-1]
            accion = random.choice(self.actions)
            nueva_pos = self.simular_accion(current_position[0], current_position[1], accion)
            # Verificar si el nuevo estado es válido
            if (0 <= nueva_pos[0] < self.rows and 0 <= nueva_pos[1] < self.cols and maze[nueva_pos[0]][nueva_pos[1]] == 0):
                other_agent_position = nueva_pos
            else:
                other_agent_position = current_position  # Mantener posición si movimiento inválido
            other_agent_positions.append(other_agent_position)
        return other_agent_position

    def train(self, maze, episodes=1000, max_steps=100):
        """
        Entrena el agente usando Q-learning.
        
        :param maze: Matriz del laberinto.
        :param episodes: Número de episodios de entrenamiento.
        :param max_steps: Número máximo de pasos por episodio.
        """
        logger.info(f"Iniciando entrenamiento para {'Policía' if self.role == 0 else 'Ladrón'}")
        for episodio in range(1, episodes + 1):
            try:
                estado_actual = self.obtener_estado_inicial(maze)
            except ValueError as e:
                logger.error(str(e))
                break  # Terminar si no hay estados iniciales válidos
            pasos = 0
            terminado = False

            # Lista para rastrear las posiciones del otro agente
            other_agent_positions = []

            for paso in range(max_steps):
                accion = self.choose_action(estado_actual)
                # Obtener siguiente estado del agente
                siguiente_estado = self.obtener_siguiente_estado(estado_actual, accion, maze)
                
                # Actualizar posición del otro agente
                other_agent_position = self.obtener_posicion_other_agent(other_agent_positions, maze)
                
                # Obtener recompensa y verificar terminación
                recompensa, terminado = self.get_reward(estado_actual, accion, maze, other_agent_position)
                
                # Actualizar Q
                self.update_Q(estado_actual, accion, recompensa, siguiente_estado)
                
                # Actualizar estado
                estado_actual = siguiente_estado
                
                # Si terminado, finalizar el episodio
                if terminado:
                    if self.role == 0:
                        logger.debug(f"Episodio {episodio}: Policía capturó al ladrón en {paso + 1} pasos.")
                    elif self.role == 1:
                        logger.debug(f"Episodio {episodio}: Ladrón fue capturado en {paso + 1} pasos.")
                    break
                
                pasos += 1
            
            # Decaimiento de epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.decay_rate
            
            # Logging cada 100 episodios
            if episodio % 100 == 0:
                logger.info(f"Episodio {episodio}/{episodes} completado.")
        
        # Generar la política después del entrenamiento
        self.politica = self.generar_politica()
        logger.info(f"Entrenamiento completado para {'Policía' if self.role == 0 else 'Ladrón'}.")

    def generar_politica(self):
        """
        Genera una política basada en la tabla Q, filtrando estados inválidos.
        """
        politica = {}
        for estado, acciones in self.Q.items():
            # Verificar que el estado está dentro de los límites
            if 0 <= estado[0] < self.rows and 0 <= estado[1] < self.cols:
                mejor_accion = max(acciones, key=acciones.get)
                politica[estado] = mejor_accion
        return politica

    def save_policy(self, archivo_q='Q_table.pkl', archivo_politica='policy.json'):
        """
        Guarda la tabla Q y la política en archivos.
        """
        # Guardar la tabla Q junto con filas y columnas
        with open(archivo_q, 'wb') as f:
            pickle.dump({
                'Q': self.Q,
                'rows': self.rows,
                'cols': self.cols
            }, f)
        logger.info(f"Tabla Q y dimensiones guardadas en {archivo_q}.")
    
        # Guardar la política
        with open(archivo_politica, 'w') as f:
            # Convertir las claves de los estados a cadenas para JSON
            politica_serializable = {str(k): v for k, v in self.politica.items()}
            json.dump(politica_serializable, f, indent=4)
        logger.info(f"Política guardada en {archivo_politica}.")

    def cargar_policy(self, archivo_q='Q_table.pkl', archivo_politica='policy.json'):
        """
        Carga la tabla Q y la política desde archivos.
        """
        # Cargar la tabla Q y dimensiones
        if os.path.exists(archivo_q):
            with open(archivo_q, 'rb') as f:
                data = pickle.load(f)
                self.Q = data['Q']
                self.rows = data['rows']
                self.cols = data['cols']
            logger.info(f"Tabla Q y dimensiones cargadas desde {archivo_q}.")
        else:
            logger.warning(f"Archivo {archivo_q} no encontrado. Inicializando tabla Q vacía.")
    
        # Cargar la política
        if os.path.exists(archivo_politica):
            with open(archivo_politica, 'r') as f:
                politica = json.load(f)
            # Convertir claves de string a tuplas y filtrar estados inválidos
            self.politica = {}
            for k, v in politica.items():
                try:
                    estado = tuple(map(int, k.strip('()').split(', ')))
                    if 0 <= estado[0] < self.rows and 0 <= estado[1] < self.cols:
                        self.politica[estado] = v
                except ValueError:
                    logger.error(f"Formato de estado inválido en política: {k}")
            logger.info(f"Política cargada desde {archivo_politica}.")
        else:
            logger.warning(f"Archivo {archivo_politica} no encontrado. Política no cargada.")
            self.politica = {}

    def get_action_from_policy(self, estado):
        """
        Obtiene la mejor acción basada en la política cargada.
        """
        return self.politica.get(estado, random.choice(self.actions))


def entrenar_Q_learning(maze, role=0, alpha=0.1, gamma=0.9, 
                        epsilon=1.0, min_epsilon=0.1, decay_rate=0.995, episodes=1000):
    """
    Entrena un agente de Q-learning.
    
    :param maze: Mapa del laberinto.
    :param role: Rol del agente (0 = Policía, 1 = Ladrón).
    :param alpha: Tasa de aprendizaje.
    :param gamma: Factor de descuento.
    :param epsilon: Tasa de exploración inicial.
    :param min_epsilon: Tasa mínima de exploración.
    :param decay_rate: Tasa de decaimiento de epsilon.
    :param episodes: Número de episodios de entrenamiento.
    :return: Agente entrenado.
    """
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0
    if rows != 7 or cols != 7:
        logger.warning(f"El mapa proporcionado no es de 7x7, sino de {rows}x{cols}. Asegúrate de que el mapa tenga las dimensiones correctas.")
    
    agent = QLearningAgent(rows, cols, role, alpha, gamma, epsilon, min_epsilon, decay_rate)
    agent.train(maze, episodes=episodes)
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
    # Necesitamos conocer las dimensiones del mapa antes de cargar la tabla Q
    archivo_q = f"Q_table_role_{role}.pkl"
    archivo_politica = f"policy_role_{role}.json"

    if os.path.exists(archivo_q):
        with open(archivo_q, 'rb') as f:
            data = pickle.load(f)
            rows = data['rows']
            cols = data['cols']
    else:
        logger.error(f"No se puede cargar el agente. El archivo {archivo_q} no existe.")
        return None

    agent = QLearningAgent(rows, cols, role)
    agent.cargar_policy(archivo_q, archivo_politica)
    return agent
