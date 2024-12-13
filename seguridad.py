# Valores proporcionados
p = 19
q = 67
n = p * q  # Modulo n
phi_n = (p - 1) * (q - 1)  # Funcion de Euler

# Clave publica
e = 5

# Clave privada
d = 713

# Criptograma
ciphertext = [1252, 1079, 319, 337, 231, 1100, 507, 1100, 231, 213, 192]

# Funcion para descifrar un bloque usando potencia modular
def decrypt_block(block, d, n):
    return pow(block, d, n)
plaintext_numbers = [decrypt_block(c, d, n) for c in ciphertext]

# Convertir a texto si los valores corresponden a ASCII legible
plaintext_chars = [chr(num) if 32 <= num <= 126 else f"[?{num}]" for num in plaintext_numbers]

print("Numeros descifrados:", plaintext_numbers)
print("Texto descifrado:", ''.join(plaintext_chars))
