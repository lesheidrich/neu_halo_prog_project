import numpy as np


class Activation:
    def __init__(self):
        self.input = None

    def forward(self, x_inp):
        self.input = x_inp
        return self.sigmoid(self.input)

    def backward(self, output_gradient, learning_rate=None):
        return np.multiply(output_gradient, self.sigmoid_prime(self.input))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-(x.astype(float))))

    def sigmoid_prime(self, x):
        s = self.sigmoid(x)
        # return np.multiply(s * (1 - s))
        return s * (1 - s)
