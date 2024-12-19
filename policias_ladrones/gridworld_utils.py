# gridworld_utils.py

import json
import random

# Clase para definir el entorno de la grilla con obstáculos
class GridWorld:
    def __init__(self, width, height, start, goal, obstacles):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start
        self.actions = ['up', 'down', 'left', 'right']

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        new_x, new_y = x, y

        if action == 'up':
            new_y = max(y - 1, 0)
        elif action == 'down':
            new_y = min(y + 1, self.height - 1)
        elif action == 'left':
            new_x = max(x - 1, 0)
        elif action == 'right':
            new_x = min(x + 1, self.width - 1)

        proposed_state = (new_x, new_y)

        if proposed_state in self.obstacles:
            reward = -0.5
            done = False
            return self.state, reward, done
        else:
            self.state = proposed_state
            if self.state == self.goal:
                reward = 1
                done = True
            else:
                reward = -0.01
                done = False
            return self.state, reward, done

# Función para inicializar los parámetros del entorno
def initialize_environment(width, height, goal, obstacles):
    start = (0, 0)
    assert start not in obstacles, "La posición de inicio no puede ser un obstáculo."
    assert goal not in obstacles, "La posición de meta no puede ser un obstáculo."

    return GridWorld(width, height, start, goal, obstacles)

# Función para transformar la matriz del laberinto en una representación descriptiva con etiquetas
def obtener_mapa_descriptivo(maze):
    """
    Transforma la matriz del laberinto en una representación descriptiva con etiquetas.
    
    Parámetros:
    maze (list of list of int): Mapa del laberinto donde 0 es pasillo y 1 es obstáculo.
    
    Retorna:
    list of list of str: Mapa descriptivo con 'S' (Inicio), 'E' (Salida), 'O' (Obstáculo), 'P' (Pasillo).
    """
    rows = len(maze)
    cols = len(maze[0])
    mapa = [['P' for _ in range(cols)] for _ in range(rows)]
    
    for i in range(rows):
        for j in range(cols):
            if maze[i][j] == 1:
                mapa[i][j] = 'O'  # Obstáculo
    
    # Definir inicio y salida
    mapa[0][0] = 'S'  # Inicio
    mapa[rows-1][cols-1] = 'E'  # Salida
    
    return mapa

def obtener_salida(mapa):
    """
    Localiza la posición de la salida 'E' en el mapa descriptivo.
    
    Parámetros:
    mapa (list of list of str): Mapa descriptivo con 'S', 'E', 'O', 'P'.
    
    Retorna:
    tuple: Coordenadas (fila, columna) de la salida.
    """
    for i, fila in enumerate(mapa):
        for j, celda in enumerate(fila):
            if celda == 'E':
                return (i, j)
    raise ValueError("Salida 'E' no encontrada en el mapa.")

def obtener_obstaculos(mapa):
    """
    Identifica todas las posiciones de los obstáculos 'O' en el mapa descriptivo.
    
    Parámetros:
    mapa (list of list of str): Mapa descriptivo con 'S', 'E', 'O', 'P'.
    
    Retorna:
    list of tuple: Lista de coordenadas (fila, columna) de los obstáculos.
    """
    obstaculos = []
    for i, fila in enumerate(mapa):
        for j, celda in enumerate(fila):
            if celda == 'O':
                obstaculos.append((i, j))
    return obstaculos
