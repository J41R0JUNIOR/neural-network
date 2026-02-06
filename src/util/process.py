import numpy as np

def preprocess_images(images):
    images = images.reshape(-1, 28*28, 1)
    return images / 255.0 

def vectorized_label(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
