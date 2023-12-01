import numpy as np


class Layer:
    def __init__(self, width, height):
        self.weights = np.random.uniform(-1, 1, size=(height, width))
        self.bias = np.zeros(height)
        self.input_vector = None
        self.backward_vector = None

    def forward(self, input_vector):
        self.input_vector = input_vector
        return self.weights @ input_vector + self.bias

    def backward(self, gradient_vector):
        self.backward_vector = np.outer(self.input_vector, gradient_vector.T)
        return self.backward_vector


class LossFunction:
    def __init__(self):
        self.input_vector = None
        self.target_vector = None
        self.backward_vector = None

    def forward(self, input_vector, target):
        self.input_vector = input_vector
        self.target_vector = target
        return sum(np.power(input_vector - target, 2))

    def backward(self):
        self.backward_vector = 2 * (self.input_vector - self.target_vector)
        return self.backward_vector
