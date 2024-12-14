import socket
import time

bluetooth_socket=None
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
    try:
        bluetooth_socket.send(message.encode())
        print(f"Comando enviado: {message}")
        time.sleep(0.1)  # Pausa breve para evitar congestión
    except Exception as e:
        print(f"Error al enviar datos: {e}")
TARGET_MAC = "00:1B:10:21:2C:34"  # Reemplaza con la dirección MAC de tu dispositivo

# Establecer conexión
bluetooth_socket = bluetooth_connect(TARGET_MAC)
for i in range(0,10):
    send_command("w")