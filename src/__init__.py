from ntpath import join
from util.load import load_mnist_images, load_mnist_labels
from Network import Network
from util.process import vectorized_label

input_path = '../mnist-dataset/'

# images
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')

# labels
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# initializing the network
network = Network([784, 30, 10])

training_data = [
    (x, vectorized_label(y))
    for x, y in 
        zip(
            load_mnist_images(training_images_filepath),
            load_mnist_labels(training_labels_filepath)
        )
]

test_data = [
    (x, y)
    for x, y in 
        zip(
            load_mnist_images(images_filepath),
            load_mnist_labels(labels_filepath)
        )
]

network.stochastic_gradient_descent(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)