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
    def __init__(self, bias, sum_inp, hidden, samples, epoch, activ, test_samples, label_y, y_test, learn_rate):
        self.B = bias
        self.nn = [sum_inp + self.B] + hidden + [len(samples[0][1])] # node num / nn layer
        self.samples = samples
        self.epoch = epoch
        # weights matrix -0.4 - 0.4
        #           num of nodes in next layer V          V num nodes in current layer
        self.wl = [np.random.random((self.nn[l + 1], self.nn[l])) * 0.8 - 0.4 for l in range(len(self.nn) - 1)]
        # layer errors
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
        self.learn_rate = learn_rate

    def train(self):
        cnt = 1
        mse = 1.0
        print("\nCOMMENCING TRAINING:")
        while mse >= 0.01 and self.epoch > 0:
            print(f"Epoch {cnt}: {mse}")
            mse = 0.0
            self.epoch -= 1
            cnt += 1
            for inp, out in self.samples:
                # each layer's neuron values
                self.nl = [np.array(list(inp) + [1.0] * self.B)]
                for l in range(len(self.nn) - 1):
                    # forward prop: w*n val
                    self.nl.append(self.acti(np.dot(self.wl[l], self.nl[l])))
                # actual - predicted
                error = out - self.nl[-1]
                # error backprop (in reverse order)
                for l in reversed(range(len(self.nn) - 1)):
                    if l == len(self.nn) - 2:
                        self.delta[l][:] = error * self.dacti(self.nl[-1])
                    else:
                        np.dot(self.delta[l + 1], self.wl[l + 1], out=self.delta[l])
                        self.delta[l] *= self.dacti(self.nl[l + 1])
                    # update weights - increased by learn rate * layer errors * neuron values
                    self.wl[l] += self.learn_rate * self.delta[l].reshape((self.nn[l + 1], 1)) * self.nl[l].reshape((1, self.nn[l]))

                mse += sum(error ** 2)
        print(f"\nRESULTS:\nEpochs: {cnt-1}\nMSE: {mse}")

    def test(self):
        for inp, out in self.test_samples:
            self.nl = [np.array(list(inp) + [1.0] * self.B)]
            # forward prop for predictions
            for l in range(len(self.nn) - 1):
                self.nl.append(self.acti(np.dot(self.wl[l], self.nl[l])))

            self.test_results.append(self.nl[-1])

        while True:
            sort = input("\nSort results by model? <y/n>: ")
            if sort == 'n' or sort == 'y':
                break

        df = pd.DataFrame(columns=["Model", "Predicted$", "Actual$", "Diff$"])
        for y in range(len(self.model_label)):
            model = self.model_label[y]
            neu = (self.test_results[y] * 56830).round(2)
            ans = (self.y_test[y] * 56830).round(2)
            diff = (abs(self.test_results[y] - self.y_test[y]) * 56830).round(2)
            row = pd.DataFrame({"Model": [model], "Predicted$": [neu], "Actual$": [ans], "Diff$": [diff]})
            df = pd.concat([df, row], ignore_index=True)

        if sort.startswith('y'):
            df = df.sort_values("Model")
        print(df[["Model", "Predicted$", "Actual$", "Diff$"]].to_string(index=False))

        print(f"Average difference: {df['Diff$'].mean().round(2)[0]}")
