from ntpath import join
from util.show import show_mnist_images

input_path = '../mnist-dataset/'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')

# just showing some MNIST images
show_mnist_images(training_images_filepath, training_labels_filepath)