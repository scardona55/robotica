# ==============================
# Importación de Librerías
# ==============================

import logging  # Para registrar eventos e información en el programa
import json  # Para procesar datos en formato JSON
import os  # Para interactuar con el sistema operativo (archivos, rutas)
import requests  # Para realizar solicitudes HTTP al servidor Flask
import time  # Para pausas y manejo del tiempo
import argparse  # Para manejar argumentos desde la línea de comandos
import math  # Para cálculos matemáticos, como distancias
import comunicacionBluetooth  # Para comunicación con el robot vía Bluetooth
from learning_agentsjuan import entrenar_Q_learning, cargar_agente  # Funciones para entrenar y cargar agentes
from gridworld_utils import obtener_mapa_descriptivo, obtener_obstaculos  # Utilidades del mapa (eliminado obtener_salida)

# ==============================
# Configuración del Logging
# ==============================

logging.basicConfig(
    level=logging.INFO,  # Nivel de registro (INFO, DEBUG, WARNING, etc.)
    format='%(asctime)s [%(levelname)s] %(message)s',  # Formato del mensaje registrado
    handlers=[
        logging.StreamHandler()  # Enviar los mensajes a la consola
    ]
)
logger = logging.getLogger(__name__)  # Configuración del objeto logger para registrar mensajes

# ==============================
# Parámetros del Servidor
# ==============================

SERVER_URL = "http://localhost:5000"  # Dirección base del servidor Flask para obtener datos

# ==============================
# Funciones para Obtener Datos del Servidor
# ==============================

def fetch_maze_data():
    """
    Obtiene el mapa del laberinto desde el servidor.
    Retorna:
        - Mapa (list of list of int) si tiene éxito.
        - None si ocurre algún error.
    """
    url = f"{SERVER_URL}/maze"  # Endpoint del servidor para el mapa
    logger.debug(f"Intentando obtener el mapa desde el endpoint: {url}")
    try:
        response = requests.get(url, timeout=10)  # Realiza la solicitud HTTP con un tiempo límite
        response.raise_for_status()  # Lanza una excepción si ocurre un error HTTP
        data = response.json()  # Convierte la respuesta en un objeto Python

        # Validar que el mapa sea una lista de listas y que contenga solo valores 0 y 1
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
    Obtiene las detecciones de formas (shapes) desde el servidor.
    Retorna:
        - Lista de detecciones si tiene éxito.
        - None si ocurre algún error.
    """
    url = f"{SERVER_URL}/detect_shapes"  # Endpoint para detecciones de shapes
    logger.debug(f"Intentando obtener las detecciones desde el endpoint: {url}")
    try:
        response = requests.get(url, timeout=10)  # Solicitud HTTP con tiempo límite
        response.raise_for_status()  # Lanza una excepción en caso de error HTTP
        data = response.json()  # Convierte la respuesta en un objeto Python
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
    """
    Clase para manejar el estado y las decisiones del robot.
    """
    def __init__(self, maze, agente_policia, agente_ladron):
        """
        Inicializa el controlador del robot con el mapa y los agentes.
        """
        self.maze = maze  # Mapa del laberinto
        self.rows = len(maze)  # Número de filas del laberinto
        self.cols = len(maze[0])  # Número de columnas del laberinto
        self.agente_policia = agente_policia  # Agente de aprendizaje del policía
        self.agente_ladron = agente_ladron  # Agente de aprendizaje del ladrón
        
        # Variables de estado del robot
        self.current_row = None  # Fila actual
        self.current_col = None  # Columna actual
        self.current_angle = 0  # Ángulo actual
        self.role = -1  # Rol (-1: desconocido, 0: policía, 1: ladrón)
        self.opponent_position = None  # Posición del oponente
    
    def actualizar_posicion(self, detections):
        """
        Actualiza la posición y rol del robot basándose en las detecciones.
        """
        robot_detections = [d for d in detections if d['shape'] in [8, 9]]  # Filtra detecciones del robot
        if not robot_detections:
            logger.warning("No se detectó el rol del robot.")
            return False

        # Actualizar las variables del estado del robot
        robot_data = robot_detections[0]
        self.current_row = robot_data['row']
        self.current_col = robot_data['col']
        self.current_angle = robot_data['angle']
        self.role = robot_data['role']

        logger.info(f"Robot detectado en fila {self.current_row}, columna {self.current_col}, ángulo {self.current_angle}°, rol {'Policía' if self.role == 0 else 'Ladrón'}")
        self.opponent_position = self.obtener_oponente(detections)  # Actualizar posición del oponente
        return True
    
    def obtener_oponente(self, detections):
        """
        Obtiene la posición del oponente.
        """
        oponentes = [d for d in detections if d['role'] != self.role and d['role'] != -1]
        if not oponentes:
            logger.warning("No se detecta al oponente.")
            return None
        return (oponentes[0]['row'], oponentes[0]['col'])
    
    def decidir_y_ejecutar_movimiento(self):
        """
        Decide y ejecuta el movimiento del robot según su rol.
        """
        if self.role == 0:  # Rol: Policía
            if not self.opponent_position:
                logger.info("Policía no tiene objetivo. Esperando detección del Ladrón...")
                return
            state = self.agente_policia.state_to_index(self.current_row, self.current_col)
            accion = self.agente_policia.get_action_from_policy(state)
        elif self.role == 1:  # Rol: Ladrón
            if not self.opponent_position:
                logger.info("Ladrón no tiene oponente detectado. Esperando detección del Policía...")
                return
            state = self.agente_ladron.state_to_index(self.current_row, self.current_col)
            accion = self.agente_ladron.get_action_from_policy(state)
        else:
            logger.warning(f"Rol desconocido: {self.role}")
            return
        
        accion_map = {
            'up': "MOVE_FORWARD",
            'down': "MOVE_BACKWARD",
            'left': "TURN_LEFT",
            'right': "TURN_RIGHT"
        }
        command = accion_map.get(accion, None)
        if command:
            comunicacionBluetooth.send_command(command)  # Envía el comando al robot
            logger.info(f"Comando enviado: {command}")
        else:
            logger.error(f"Acción desconocida '{accion}' para el comando.")

# ==============================
# Funciones para Entrenar y Cargar Políticas
# ==============================

def train_policies_if_needed(maze):
    """
    Entrena o carga las políticas para Policía y Ladrón según sea necesario.
    """
    policia_policy_file = 'policy_role_0.json'  # Archivo de política para Policía
    ladron_policy_file = 'policy_role_1.json'  # Archivo de política para Ladrón
    policia_q_file = 'Q_table_role_0.pkl'  # Tabla Q para Policía
    ladron_q_file = 'Q_table_role_1.pkl'  # Tabla Q para Ladrón

    agentes = {}
    if not os.path.exists(policia_q_file) or not os.path.exists(policia_policy_file):
        logger.info("Entrenando política para Policía...")
        agente_policia = entrenar_Q_learning(maze=maze, role=0, episodes=1000)
        agentes[0] = agente_policia
    else:
        logger.info("Cargando política existente para Policía.")
        agentes[0] = cargar_agente(role=0)
    
    if not os.path.exists(ladron_q_file) or not os.path.exists(ladron_policy_file):
        logger.info("Entrenando política para Ladrón...")
        agente_ladron = entrenar_Q_learning(maze=maze, role=1, episodes=1000)
        agentes[1] = agente_ladron
    else:
        logger.info("Cargando política existente para Ladrón.")
        agentes[1] = cargar_agente(role=1)
    
    return agentes

# ==============================
# Función Principal
# ==============================

def main():
    """
    Función principal que coordina todo el programa.
    """
    parser = argparse.ArgumentParser(description='Navegación de Robot con Q-learning')
    parser.add_argument('--algo', type=str, choices=['qlearning'], default='qlearning',
                        help='Selecciona el algoritmo de aprendizaje: "qlearning" (por defecto: qlearning)')
    args = parser.parse_args()
    algoritmo = args.algo.lower()

    logger.info(f"Algoritmo seleccionado: {algoritmo.upper()}")

    # Obtener el mapa desde el servidor
    maze = fetch_maze_data()
    if maze is None:
        logger.error("No se pudo obtener el mapa desde el endpoint. Terminando el programa.")
        return

    rows = len(maze)
    cols = len(maze[0]) if rows > 0 else 0

    if rows == 0 or cols == 0:
        logger.error("El mapa cargado está vacío. Terminando el programa.")
        return

    if not all(len(row) == cols for row in maze):
        logger.error("El mapa cargado no es rectangular. Todas las filas deben tener la misma longitud.")
        return

    logger.info(f"Mapa con dimensiones {rows}x{cols} cargado.")

    mapa = obtener_mapa_descriptivo(maze)
    obstaculos = obtener_obstaculos(mapa)  # Eliminado obtener_salida

    agentes = train_policies_if_needed(maze)
    agente_policia = agentes.get(0)
    agente_ladron = agentes.get(1)

    robot = RobotController(maze, agente_policia, agente_ladron)

    try:
        while True:
            detections = fetch_detect_shapes()
            if detections is None:
                logger.warning("No se pudieron obtener las detecciones. Reintentando en 2 segundos...")
                time.sleep(2)
                continue

            if not robot.actualizar_posicion(detections):
                logger.warning("No se detectó el robot en las detecciones. Reintentando en 2 segundos...")
                time.sleep(2)
                continue

            robot.decidir_y_ejecutar_movimiento()
            time.sleep(1)  # Ajusta este tiempo según la velocidad del robot

    except KeyboardInterrupt:
        logger.info("Interrupción por teclado. Cerrando programa...")

    finally:
        logger.info("Conexión Bluetooth cerrada y recursos liberados.")

if __name__ == "__main__":
    main()
