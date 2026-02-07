import numpy as np
from util.sigmoid import sigmoid, derivative_sigmoid;

class Network(object):

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            len_test_data = len(test_data)

        len_training_data = len(training_data)

        for epoch in range(epochs):
            np.random.shuffle(training_data)

            mini_batches = [
                training_data[k:k + mini_batch_size] 
                for k in range(0, len_training_data, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update_network(mini_batch, eta)
            
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len_test_data}")
            else:
                print(f"Epoch {epoch} complete")
    
    def update_network(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.back_propagation(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def back_propagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        delta_c = self.cost_derivative(activations[-1], y) * derivative_sigmoid(zs[-1])
        nabla_b[-1] = delta_c
        nabla_w[-1] = np.dot(delta_c, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = derivative_sigmoid(z)
            delta_c = np.dot(self.weights[-l + 1].transpose(), delta_c) * sp
            nabla_b[-l] = delta_c
            nabla_w[-l] = np.dot(delta_c, activations[-l - 1].transpose())
        
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        correct_predictions = 0
        
        for x, y in test_data:
            predicted_label = np.argmax(self.feedforward(x))
            if predicted_label == y:
                correct_predictions += 1
        return correct_predictions

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a