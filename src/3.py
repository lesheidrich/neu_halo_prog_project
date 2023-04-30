import numpy, random


def sigmoid_activation_func(x):    return 1.0 / (1.0 + numpy.exp(-x))


def sigmoid_activation_func_derivative(y):   return y * (1.0 - y)


# def acti(x):     return numpy.tanh(x)
# def dacti(y):    return 1.0 - y**2

NI, NH, NO, B = 2, 4, 3, 1
w1 = numpy.random.random((NH, NI + B)) * 0.8 - 0.4
# w1 = numpy.linspace(-0.4,0.4,NH*(NI+B)).reshape((NH, NI+B))
# w1 = numpy.ones((NI+1, NH))*0.2
# w1 = numpy.array([ [(h*1.0)/NH]*(NI+1) for h in range(NH) ])

w2 = numpy.random.random((NO, NH)) * 0.8 - 0.4
# w2 = numpy.linspace(-0.4,0.4,NO*NH).reshape((NO, NH))
# w2 = numpy.ones((NH, NO))*0.2
# w2 = numpy.array([ [(o*1.0)/NO]*NH for o in range(NO) ])

samples = [
    [[0, 0], [0, 0, 0]], [[0, 1], [0, 1, 1]], [[1, 0], [0, 1, 1]], [[1, 1], [1, 1, 0]]
]
# samples = [[[0, 0], [0, 0, 0, 1]], [[0, 1], [0, 1, 1, 0]], [[1, 0], [0, 1, 1, 0]], [[1, 1], [1, 1, 0, 1]]]



for cnt in range(10000):
    sumerr = 0.0
    for inp, out in samples:
        x = numpy.array(inp + [1.0] * B)
        h = sigmoid_activation_func(numpy.dot(w1, x))
        y = sigmoid_activation_func(numpy.dot(w2, h))
        error = out - y
        deltao = error * sigmoid_activation_func_derivative(y)
        deltah = numpy.dot(deltao, w2) * sigmoid_activation_func_derivative(h)
        w2 += 0.5 * deltao.reshape((NO, 1)) * h.reshape((1, NH))
        w1 += 0.5 * deltah.reshape((NH, 1)) * x.reshape((1, NI + B))
        sumerr += sum(error ** 2)
    if sumerr < 0.01:
        break
print(cnt, sumerr)
#        AND
#       0   1
#    0  0   0
#    1  0   1
#        OR
#       0   1
#    0  0   1
#    1  1   1
#        XOR
#       0   1
#    0  0   1
#    1  1   0
