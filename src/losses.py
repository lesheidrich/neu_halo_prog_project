import numpy as np


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

import numpy


# available activation functions
def activation_tanh(x):     return numpy.tanh(x)  # (-1..1)


def dactivation_tanh(x):    return 1.0 - x ** 2


def activation_sigmoid(x):  return 1.0 / (1.0 + numpy.exp(-x))  # (0..1)


def dactivation_sigmoid(x): return x * (1.0 - x)


# acti, dacti = activation_sigmoid, dactivation_sigmoid
acti, dacti = activation_tanh, dactivation_tanh

# samples = [[[0, 0], [0, 0]], [[0, 1], [0, 1]], [[1, 0], [0, 1]], [[1, 1], [1, 1]]]   # AND ĂŠs OR
samples = [[[0, 0], [0, 0, 0]], [[0, 1], [0, 1, 1]], [[1, 0], [0, 1, 1]], [[1, 1], [1, 1, 0]]]  # AND, OR ĂŠs XOR

B = 1
# nn = [2+B, 4, 3, 3]
nn = [len(samples[0][0]) + B, 4, 3, len(samples[0][1])]
wl = [numpy.random.random((nn[l + 1], nn[l])) * 0.8 - 0.4 for l in range(len(nn) - 1)]
# wl = [numpy.linspace(-1,1,nn[l]*nn[l+1]).reshape((nn[l+1], nn[l])) for l in range(len(nn)-1)]
delta = [numpy.zeros((nn[l + 1])) for l in range(len(nn) - 1)]

epoch = 0
sumerr = 1.0
while sumerr >= 0.01 and epoch <= 10000:
    sumerr = 0.0
    epoch += 1
    # for inp, out in random.sample(samples, len(samples)):
    for inp, out in samples:
        nl = [numpy.array(inp + [1.0] * B)]
        for l in range(len(nn) - 1):
            nl.append(acti(numpy.dot(wl[l], nl[l])))
        error = out - nl[-1]
        # delta = [None for _ in range(len(nn)-1)]
        for l in reversed(range(len(nn) - 1)):
            if l == len(nn) - 2:
                # delta[l] = error*dacti(nl[-1])
                delta[l][:] = error * dacti(nl[-1])
            else:
                # delta[l] = numpy.dot(delta[l+1],wl[l+1])*dacti(nl[l+1])
                numpy.dot(delta[l + 1], wl[l + 1], out=delta[l])
                delta[l] *= dacti(nl[l + 1])

            wl[l] += 0.5 * delta[l].reshape((nn[l + 1], 1)) * nl[l].reshape((1, nn[l]))

        sumerr += sum(error ** 2)
    # print (epoch,sumerr)
print(epoch, sumerr)
def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)
