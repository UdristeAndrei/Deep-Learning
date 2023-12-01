import numpy as np


class ReLu:
    def __init__(self):
        self.input_vector = None
        self.backward_vector = None

    def forward(self, input_vector):
        self.input_vector = input_vector
        return np.maximum(input_vector, 0)

    def backward(self, gradient_vector):
        self.backward_vector = np.multiply([0 if x <= 0 else 1 for x in self.input_vector], gradient_vector)
        return self.backward_vector
