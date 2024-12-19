# main.py

import logging
import json
import os
import requests
import time
import argparse
import math
import comunicacionBluetooth  # Asegúrate de que este módulo esté correctamente implementado
from learning_agents import entrenar_Q_learning, cargar_agente
from gridworld_utils import obtener_mapa_descriptivo, obtener_salida, obtener_obstaculos

# ==============================
# Configuración del Logging
# ==============================

logging.basicConfig(
    level=logging.INFO,  # Cambia a DEBUG para más detalles
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==============================
# Parámetros del Servidor
# ==============================

SERVER_URL = "http://localhost:5000"  # URL base del servidor Flask

# ==============================
# Funciones para Obtener Datos del Servidor
# ==============================

def fetch_maze_data():
    """
    Realiza una solicitud GET al endpoint para obtener el mapa del laberinto.
    Retorna:
        list of list of int: Mapa del laberinto donde 0 es pasillo y 1 es obstáculo.
        None: Si ocurre un error al obtener o procesar el mapa.
    """
    url = f"{SERVER_URL}/maze"
    logger.debug(f"Intentando obtener el mapa desde el endpoint: {url}")
    try:
        response = requests.get(url, timeout=10)  # Añadido timeout para evitar bloqueos
        response.raise_for_status()  # Lanza una excepción si hay un error HTTP
        data = response.json()  # Convierte la respuesta JSON a una lista
        logger.debug(f"Respuesta del endpoint: {data}")
        
        # Verificar que el mapa sea una lista de listas y que contenga solo 0 y 1
        if isinstance(data, list) and all(isinstance(row, list) for row in data):
            for row in data:
                if not all(cell in [0, 1] for cell in row):
                    logger.error("El mapa contiene valores inválidos. Solo se permiten 0 y 1.")
                    return None
            logger.info("Mapa cargado correctamente desde el endpoint.")
            return data
        else:
            logger.error("El formato del mapa es inválido. Debe ser una lista de listas.")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ocurrió un error al obtener el mapa: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON: {e}")
        return None

def fetch_detect_shapes():
    """
    Realiza una solicitud GET al endpoint para obtener las detecciones de shapes.
    Retorna:
        list of dict: Lista de detecciones con información relevante.
        None: Si ocurre un error al obtener o procesar las detecciones.
    """
    url = f"{SERVER_URL}/detect_shapes"
    logger.debug(f"Intentando obtener las detecciones desde el endpoint: {url}")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        logger.debug(f"Respuesta del endpoint detect_shapes: {data}")
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"Ocurrió un error al obtener las detecciones: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON: {e}")
        return None

# ==============================
# Clase para Controlar el Estado del Robot
# ==============================

class RobotController:
    def __init__(self, maze, agente_policia, agente_ladron):
        self.maze = maze
        self.rows = len(maze)
        self.cols = len(maze[0])
        self.agente_policia = agente_policia
        self.agente_ladron = agente_ladron
        
        # Información de posición y navegación
        self.current_row = None
        self.current_col = None
        self.current_angle = 0  # Inicializar con un ángulo predeterminado
        self.role = -1  # -1 = Desconocido, 0 = Policía, 1 = Ladrón
        
        # Posición del oponente
        self.opponent_position = None
    
    def actualizar_posicion(self, detections):
        """
        Actualiza la posición y rol del robot basado en las detecciones.
        """
        # Buscar el marcador correspondiente al robot
        robot_detections = [d for d in detections if d['shape'] in [8, 9]]
        if not robot_detections:
            logger.warning("No se detectó el rol del robot.")
            return False

        robot_data = robot_detections[0]
        self.current_row = robot_data['row']
        self.current_col = robot_data['col']
        self.current_angle = robot_data['angle']
        self.role = robot_data['role']

        logger.info(f"Robot detectado en fila {self.current_row}, columna {self.current_col}, ángulo {self.current_angle}°, rol {'Policía' if self.role == 0 else 'Ladrón'}")
        
        # Actualizar la posición del oponente
        self.opponent_position = self.obtener_oponente(detections)
        
        return True
    
    def obtener_oponente(self, detections):
        """
        Obtiene la posición del oponente basado en las detecciones.
        """
        oponentes = [d for d in detections if d['role'] != self.role and d['role'] != -1]
        if not oponentes:
            logger.warning("No se detecta al oponente.")
            return None
        # Asumir que hay solo un oponente
        return (oponentes[0]['row'], oponentes[0]['col'])
    
    def decidir_y_ejecutar_movimiento(self):
        """
        Decide y ejecuta el siguiente movimiento basado en el rol del robot.
        """
        if self.role == 0:
            # Rol: Policía
            if not self.opponent_position:
                logger.info("Policía no tiene objetivo. Esperando detección del Ladrón...")
                return
            target_position = self.opponent_position
            state = self.agente_policia.state_to_index(self.current_row, self.current_col)
            accion = self.agente_policia.get_action_from_policy(state)
            accion_map = {
                'up': "MOVE_FORWARD",
                'down': "MOVE_BACKWARD",
                'left': "TURN_LEFT",
                'right': "TURN_RIGHT"
            }
            command = accion_map.get(accion, None)
            if command:
                comunicacionBluetooth.send_command(command)
                logger.info(f"Comando enviado (Policía): {command}")
        elif self.role == 1:
            # Rol: Ladrón
            target_position = self.calcular_casilla_mas_alejada()
            state = self.agente_ladron.state_to_index(self.current_row, self.current_col)
            accion = self.agente_ladron.get_action_from_policy(state)
            accion_map = {
                'up': "MOVE_FORWARD",
                'down': "MOVE_BACKWARD",
                'left': "TURN_LEFT",
                'right': "TURN_RIGHT"
            }
            command = accion_map.get(accion, None)
            if command:
                comunicacionBluetooth.send_command(command)
                logger.info(f"Comando enviado (Ladrón): {command}")
        else:
            logger.warning(f"Rol desconocido: {self.role}")
    
    def calcular_casilla_mas_alejada(self):
        """
        Calcula la casilla más alejada del Policía.
        """
        if not self.opponent_position:
            logger.warning("No se puede calcular la casilla más alejada sin la posición del Policía.")
            return (self.current_row, self.current_col)
        
        max_distancia = -1
        casilla_mas_alejada = (self.current_row, self.current_col)
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == 0:
                    distancia = math.hypot(self.opponent_position[0] - i, self.opponent_position[1] - j)
                    if distancia > max_distancia:
                        max_distancia = distancia
                        casilla_mas_alejada = (i, j)
        
        logger.info(f"Casilla más alejada calculada: {casilla_mas_alejada} con distancia {max_distancia}")
        return casilla_mas_alejada

# ==============================
# Funciones para Entrenar y Cargar Políticas
# ==============================

def train_policies_if_needed(maze, salida):
    """
    Entrena las políticas para Policía y Ladrón si no existen archivos de política.
    Retorna los agentes entrenados.
    """
    policia_policy_file = 'policy_role_0.json'
    ladron_policy_file = 'policy_role_1.json'
    policia_q_file = 'Q_table_role_0.pkl'
    ladron_q_file = 'Q_table_role_1.pkl'

    agentes = {}
    # Entrenar para Policía
    if not os.path.exists(policia_q_file) or not os.path.exists(policia_policy_file):
        logger.info("Entrenando política para Policía...")
        agente_policia = entrenar_Q_learning(
            maze=maze,
            salida=salida,
            role=0,
            episodes=1000
        )
        agentes[0] = agente_policia
    else:
        logger.info("Cargando política existente para Policía.")
        agente_policia = cargar_agente(role=0)
        agentes[0] = agente_policia

    # Entrenar para Ladrón
    if not os.path.exists(ladron_q_file) or not os.path.exists(ladron_policy_file):
        logger.info("Entrenando política para Ladrón...")
        agente_ladron = entrenar_Q_learning(
            maze=maze,
            salida=salida,
            role=1,
            episodes=1000
        )
        agentes[1] = agente_ladron
    else:
        logger.info("Cargando política existente para Ladrón.")
        agente_ladron = cargar_agente(role=1)
        agentes[1] = agente_ladron

    return agentes

# ==============================
# Función Principal
# ==============================

def main():
    # Manejar argumentos de línea de comandos para seleccionar el algoritmo
    parser = argparse.ArgumentParser(description='Navegación de Robot con Q-learning y SARSA')
    parser.add_argument('--algo', type=str, choices=['qlearning', 'sarsa'], default='qlearning',
                        help='Selecciona el algoritmo de aprendizaje: "qlearning" o "sarsa" (por defecto: qlearning)')
    args = parser.parse_args()
    algoritmo = args.algo.lower()

    logger.info(f"Algoritmo seleccionado: {algoritmo.upper()}")

    # Obtener el mapa desde el servidor
    maze = fetch_maze_data()
    if maze is None:
        logger.error("No se pudo obtener el mapa desde el endpoint. Terminando el programa.")
        return

    # Actualizar ROWS y COLS según el mapa cargado
    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0

    if rows == 0 or cols == 0:
        logger.error("El mapa cargado está vacío. Terminando el programa.")
        return

    # Verificar que todas las filas tengan la misma longitud
    if not all(len(row) == cols for row in maze):
        logger.error("El mapa cargado no es rectangular. Todas las filas deben tener la misma longitud.")
        return

    logger.info(f"Mapa con dimensiones {rows}x{cols} cargado.")

    # Obtener salida y obstáculos
    mapa = obtener_mapa_descriptivo(maze)
    salida = obtener_salida(mapa)
    obstaculos = obtener_obstaculos(mapa)

    # Entrenar o cargar las políticas
    agentes = train_policies_if_needed(maze, salida)
    agente_policia = agentes.get(0)
    agente_ladron = agentes.get(1)

    # Crear el controlador del robot
    robot = RobotController(maze, agente_policia, agente_ladron)

    try:
        while True:
            # Obtener las detecciones desde el servidor
            detections = fetch_detect_shapes()
            if detections is None:
                logger.warning("No se pudieron obtener las detecciones. Reintentando en 2 segundos...")
                time.sleep(2)
                continue

            # Actualizar posición y rol del robot
            if not robot.actualizar_posicion(detections):
                logger.warning("No se detectó el robot en las detecciones. Reintentando en 2 segundos...")
                time.sleep(2)
                continue

            # Decidir y ejecutar el siguiente movimiento
            robot.decidir_y_ejecutar_movimiento()

            # Esperar un breve período antes de la siguiente iteración
            time.sleep(1)  # Ajusta este tiempo según la velocidad de tu robot

    except KeyboardInterrupt:
        logger.info("Interrupción por teclado. Cerrando programa...")

    finally:
        # Libera recursos
        #comunicacionBluetooth.close_connection()
        logger.info("Conexión Bluetooth cerrada y recursos liberados.")

if __name__ == "__main__":
    main()
