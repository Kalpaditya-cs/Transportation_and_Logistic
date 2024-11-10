from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import os

# AES key and IV (these should be securely stored in environment variables)
key = os.urandom(32)  # Generate AES-256 key
iv = os.urandom(16)   # Generate AES IV

# Function to encrypt the model
def encrypt_model(model_path, encrypted_model_path):
    with open(model_path, 'rb') as f:
        model_data = f.read()

    encryptor = Cipher(
        algorithms.AES(key),
        modes.CFB(iv),
        backend=default_backend()
    ).encryptor()

    encrypted_data = encryptor.update(model_data) + encryptor.finalize()

    with open(encrypted_model_path, 'wb') as f_enc:
        f_enc.write(iv + encrypted_data)  # Store IV and encrypted model data

# Function to decrypt the model
def decrypt_model(encrypted_model_path):
    with open(encrypted_model_path, 'rb') as f_enc:
        iv = f_enc.read(16)  # First 16 bytes for IV
        encrypted_data = f_enc.read()

    decryptor = Cipher(
        algorithms.AES(key),
        modes.CFB(iv),
        backend=default_backend()
    ).decryptor()

    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

    return decrypted_data

# Encrypt YOLOv10 model
encrypt_model('main.py', 'main.enc')

# Decrypt before loading
decrypted_model_data = decrypt_model('main.enc')
with open('main.pt', 'wb') as f:
    f.write(decrypted_model_data)
