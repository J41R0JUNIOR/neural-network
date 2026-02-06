import struct
import numpy as np
from util.process import preprocess_images

def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number inválido: {magic}"

        data = np.frombuffer(
            f.read(), dtype=np.uint8
        )
    
    return preprocess_images(data)

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic number inválido: {magic}"

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels
