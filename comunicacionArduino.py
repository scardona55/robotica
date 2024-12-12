import time

# Simulación de conexión serial (modo ensayo)
SIMULATE_CONNECTION = True

if not SIMULATE_CONNECTION:
    import serial

# Configuración del puerto serial
PORT = "COM5"  # Cambia esto según el puerto asignado al Bluetooth
BAUD_RATE = 115200  # Velocidad de comunicación

# Crear la conexión serial solo si no está en modo simulación
bt_connection = None
if not SIMULATE_CONNECTION:
    try:
        bt_connection = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Esperar a que el módulo Bluetooth esté listo
    except Exception as e:
        raise RuntimeError(f"Error al conectar con el puerto {PORT}: {e}")

# Función para enviar un comando
def send_command(command):
    """
    Envía un comando al Arduino a través de la conexión serial.
    
    Args:
        command (str): El comando a enviar.
    """
    if SIMULATE_CONNECTION:
        return f"[SIMULACIÓN] Comando enviado: {command}"
    elif bt_connection and bt_connection.is_open:
        bt_connection.write(command.encode('utf-8'))
        time.sleep(0.1)  # Pausa breve para evitar congestión
        return f"Comando enviado: {command}"
    else:
        raise ConnectionError("La conexión serial no está abierta")

# Función para enviar comandos desde otro script
def send_command_external(command):
    """
    Envía un comando desde un script externo al Arduino.
    
    Args:
        command (str): El comando a enviar.
    """
    return send_command(command)

# Función para leer datos del Arduino
def read_from_arduino():
    """
    Lee datos del Arduino desde la conexión serial.

    Returns:
        str: Los datos recibidos o None si no hay datos disponibles.
    """
    if SIMULATE_CONNECTION:
        return "[SIMULACIÓN] Respuesta del Arduino"
    elif bt_connection and bt_connection.is_open and bt_connection.in_waiting > 0:
        return bt_connection.readline().decode('utf-8').strip()
    return None

# Cerrar la conexión al finalizar
def close_connection():
    """
    Cierra la conexión serial con el Arduino.
    """
    if not SIMULATE_CONNECTION and bt_connection and bt_connection.is_open:
        bt_connection.close()
