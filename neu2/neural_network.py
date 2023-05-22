import numpy as np


class NeuralNetwork:
    def __init__(self, network, loss, loss_prime):
        self.network = network
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, inp):
        output = list(inp)
        for layer in self.network:
            output = layer.forward(output)
        return output

    def train(self, x_train, y_train, epochs=1000, learning_rate=0.01, verbose=True):
        for e in range(epochs):
            error = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # error
                error += self.loss(y, output)

                # print("y", np.shape(y))
                # print("output", np.shape(output))

                # backward
                grad = self.loss_prime(y, output)

                # print("grad", np.shape(grad))

                for layer in reversed(self.network):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)

            if verbose:
                print(f"{e + 1}/{epochs}, error={error}")
