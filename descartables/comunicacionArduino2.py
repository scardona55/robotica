# comunicacionArduino.py
import serial
import time
import threading

# ==============================
# Configuración del Modo de Simulación
# ==============================

SIMULATION_MODE = True  # Establece en True para activar el modo de simulación

# ==============================
# Configuración de la Comunicación Serial
# ==============================

# Configuración del puerto serial
PORT = "COM5"  # Cambia esto según el puerto asignado al Bluetooth
BAUD_RATE = 115200  # Velocidad de comunicación

# Variable global para la conexión serial
bt_connection = None

if not SIMULATION_MODE:
    # Crear la conexión serial solo si no está en modo de simulación
    try:
        bt_connection = serial.Serial(PORT, BAUD_RATE, timeout=1)
        print(f"Conectado al puerto {PORT}")
        time.sleep(2)  # Esperar a que el módulo Bluetooth esté listo
    except Exception as e:
        print(f"Error al conectar con el puerto {PORT}: {e}")
        bt_connection = None
else:
    print("Modo de Simulación Activado. No se establecerá conexión serial real.")

# ==============================
# Funciones de Comunicación Serial
# ==============================

def send_command(command):
    """
    Envía un comando al Arduino.
    En modo de simulación, solo imprime el comando.
    """
    if SIMULATION_MODE:
        print(f"[Simulación] Comando enviado: {command}")
        # Simular una respuesta después de enviar un comando
        simulate_response(command)
    else:
        if bt_connection and bt_connection.is_open:
            try:
                bt_connection.write(command.encode('utf-8'))  # Enviar el comando como bytes
                print(f"Comando enviado: {command}")
                time.sleep(0.1)  # Pausa breve para evitar congestión
            except Exception as e:
                print(f"Error al enviar el comando: {e}")
        else:
            print("La conexión serial no está abierta")

def read_from_arduino():
    """
    Lee datos del Arduino.
    En modo de simulación, retorna una respuesta simulada.
    """
    if SIMULATION_MODE:
        # En modo de simulación, podrías retornar respuestas predefinidas o aleatorias
        # Por simplicidad, no retornamos nada aquí
        return None
    else:
        if bt_connection and bt_connection.is_open and bt_connection.in_waiting > 0:
            try:
                data = bt_connection.readline().decode('utf-8').strip()
                return data
            except Exception as e:
                print(f"Error al leer del Arduino: {e}")
        return None

def close_connection():
    """
    Cierra la conexión serial si está abierta.
    En modo de simulación, no realiza ninguna acción.
    """
    if not SIMULATION_MODE:
        if bt_connection and bt_connection.is_open:
            bt_connection.close()
            print("Conexión serial cerrada.")
    else:
        print("Modo de Simulación: No se requiere cerrar conexión serial.")

# ==============================
# Función de Simulación de Respuestas
# ==============================

def simulate_response(command):
    """
    Simula una respuesta del Arduino tras recibir un comando.
    """
    # Diccionario de respuestas simuladas
    responses = {
        'w': "Movimiento Adelante",
        's': "Movimiento Atrás",
        'a': "Giro Izquierda",
        'd': "Giro Derecha",
        'x': "Detenido",
        'q': "Cerrando conexión de simulación"
    }
    # Simular un retardo antes de la respuesta
    def delayed_response():
        time.sleep(0.5)  # Retardo de 0.5 segundos
        response = responses.get(command, "Comando no reconocido en simulación")
        print(f"[Simulación] Arduino dice: {response}")
    threading.Thread(target=delayed_response, daemon=True).start()
