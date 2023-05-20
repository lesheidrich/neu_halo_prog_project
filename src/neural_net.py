import numpy as np
import pandas as pd


def sigmoid_act(x):
    return 1.0 / (1.0 + np.exp(-x.astype(float)))  # (0..1)


def dx_sigmoid_act(x):
    return x * (1.0 - x)


def tan_h_act(x):
    return np.tanh(x.astype(float))  # (-1..1)


def dx_tan_h(x):
    return 1.0 - x.astype(float) ** 2


class NeuralNet:
    def __init__(self, bias, sum_inp, hidden, samples, epoch, activ, test_samples, label_y, y_test):
        self.B = bias
        self.nn = [sum_inp + self.B] + hidden + [len(samples[0][1])]
        self.samples = samples
        self.epoch = epoch
        self.wl = [np.random.random((self.nn[l + 1], self.nn[l])) * 0.8 - 0.4 for l in range(len(self.nn) - 1)]
        self.delta = [np.zeros((self.nn[l + 1])) for l in range(len(self.nn) - 1)]
        self.nl = []
        if activ == "sig":
            self.acti, self.dacti = sigmoid_act, dx_sigmoid_act
        elif activ == "tanh":
            self.acti, self.dacti = tan_h_act, dx_tan_h
        self.test_samples = test_samples
        self.test_results = []
        self.model_label = label_y
        self.y_test = y_test

    def train(self):
        cnt = 1
        mse = 1.0
        while mse >= 0.01 and self.epoch > 0:
            print(f"Epoch {cnt}: {mse}")
            mse = 0.0
            self.epoch -= 1
            cnt += 1
            for inp, out in self.samples:
                self.nl = [np.array(list(inp) + [1.0] * self.B)]
                for l in range(len(self.nn) - 1):
                    self.nl.append(self.acti(np.dot(self.wl[l], self.nl[l])))
                error = out - self.nl[-1]
                for l in reversed(range(len(self.nn) - 1)):
                    if l == len(self.nn) - 2:
                        self.delta[l][:] = error * self.dacti(self.nl[-1])
                    else:
                        np.dot(self.delta[l + 1], self.wl[l + 1], out=self.delta[l])
                        self.delta[l] *= self.dacti(self.nl[l + 1])

                    self.wl[l] += 0.5 * self.delta[l].reshape((self.nn[l + 1], 1)) * self.nl[l].reshape((1, self.nn[l]))

                mse += sum(error ** 2)
        print(f"\nTraining results:\nEpochs: {cnt}\nMSE: {mse}")

    def test(self):
        for inp, out in self.test_samples:
            self.nl = [np.array(list(inp) + [1.0] * self.B)]
            for l in range(len(self.nn) - 1):
                self.nl.append(self.acti(np.dot(self.wl[l], self.nl[l])))

            self.test_results.append(self.nl[-1])

        df = pd.DataFrame(columns=["Model", "Neu", "Ans", "Diff"])
        for y in range(len(self.model_label)):
            model = self.model_label[y]
            neu = self.test_results[y]
            ans = self.y_test[y]
            diff = (abs(self.test_results[y] - self.y_test[y]) * 100).round(4)
            row = pd.DataFrame({"Model": [model], "Neu": [neu], "Ans": [ans], "Diff": [diff]})
            df = pd.concat([df, row], ignore_index=True)

        print(df[["Model", "Neu", "Ans", "Diff"]].to_string(index=False))