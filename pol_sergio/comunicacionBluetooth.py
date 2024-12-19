# comunicacionBluetooth.py

import socket
import time

bluetooth_socket = None

# Función para conectar con Bluetooth
def bluetooth_connect(mac_address):
    """
    Conexión Bluetooth de bajo nivel usando sockets
    """
    try:
        # Crear un socket Bluetooth
        sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)

        # Puerto por defecto (puede variar según el dispositivo)
        port = 1

        # Intentar conectar
        print(f"Conectando a {mac_address}...")
        sock.connect((mac_address, port))
        print("Conexión establecida exitosamente")

        return sock

    except Exception as e:
        print(f"Error de conexión: {e}")
        return None

# Función para enviar datos a través del socket Bluetooth
def send_command(message):
    """
    Enviar datos a través del socket Bluetooth
    """
    global bluetooth_socket
    try:
        if bluetooth_socket:
            bluetooth_socket.send(message.encode())
            print(f"Comando enviado: {message}")
            time.sleep(0.1)  # Pausa breve para evitar congestión
        else:
            print("No hay conexión Bluetooth establecida.")
    except Exception as e:
        print(f"Error al enviar datos: {e}")

# Función para cerrar la conexión Bluetooth
def close_connection():
    global bluetooth_socket
    if bluetooth_socket:
        bluetooth_socket.close()
        print("Conexión Bluetooth cerrada.")
        bluetooth_socket = None
