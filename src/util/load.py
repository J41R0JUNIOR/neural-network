import struct
import numpy as np

def load_mnist_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Magic number inválido: {magic}"

        images = np.frombuffer(
            f.read(), dtype=np.uint8
        ).reshape(num, rows, cols)

    return images

def load_mnist_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Magic number inválido: {magic}"

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels

