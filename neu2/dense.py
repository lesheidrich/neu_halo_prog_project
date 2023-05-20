import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.input = None
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, x_inp):
        self.input = x_inp
        print("w shape", np.shape(self.weights))
        print("input", np.shape(self.input))
        # if np.shape(np.asmatrix(self.input))[0] == 1:
        #     return np.dot(np.asmatrix(self.weights), np.asmatrix(self.input).transpose()) + np.asmatrix(self.bias)
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        # print("learning rate", learning_rate)
        # print("output_gradient", output_gradient)
        # print("learning rate", type(learning_rate))
        # print(type(output_gradient))
        # print(type(output_gradient[0]))
        # print(type(output_gradient[0][0]))
        # print(self.bias)
        self.bias -= learning_rate * output_gradient.astype(float)
        return input_gradient

