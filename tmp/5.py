import math, numpy

import numpy as np
import pandas as pd


# available activation functions
def activation_tanh(x):     return numpy.tanh(x.astype(float))  # (-1..1)


def dactivation_tanh(x):    return 1.0 - x ** 2


def activation_sigmoid(x):  return 1.0 / (1.0 + numpy.exp(-x.astype(float)))  # (0..1)


def dactivation_sigmoid(x): return x * (1.0 - x)


def normalize(col: list) -> list:
    """
    :param col: int based list of column values to normalize
    :return: normalized (0-1) list reshaped to vertical
    """
    arr = (col - np.min(col)) / (np.max(col) - np.min(col))
    return arr.reshape(-1, 1)


def one_hot_encode(col: list) -> list:
    """
    :param col: list of values to encode
    :return: list of one-hot encoded values
    """
    col_unique = list(set(col))
    col_train = []
    for value in col:
        one_hot = [1 if value == v else 0 for v in col_unique]
        col_train.append(one_hot)
    return col_train


def sigmoid_activation_func(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_activation_func_derivative(y):
    return y * (1.0 - y)


if __name__ == '__main__':
    df = pd.read_csv('../dataset/honda_sell_data.csv')
    filtered_cols = df[['Year', 'Model', 'Price', 'Exterior_Color', 'Drivetrain', 'Fuel_Type',
                        'Engine', 'Mileage', 'Seller_Type']]
    filtered_cols = filtered_cols[filtered_cols['Price'] != 'Not Priced']

    year = filtered_cols['Year'].to_numpy()
    price = filtered_cols['Price'].to_numpy()
    mileage = filtered_cols['Mileage'].to_numpy()
    model = filtered_cols['Model'].to_numpy()  # lehet le kell csokkenteni
    color = filtered_cols['Exterior_Color'].to_numpy()
    drivetrain = filtered_cols['Drivetrain'].to_numpy()
    engine = filtered_cols['Engine'].to_numpy()
    fuel_type = filtered_cols['Fuel_Type'].to_numpy()
    seller = filtered_cols['Seller_Type'].to_numpy()

    # min-max
    price = [int(p.replace("$", "").replace(",", "")) for p in price]
    for i, mile in enumerate(mileage):
        if isinstance(mile, float) or mile == '–':
            mileage[i] = 0
        elif isinstance(mile, str) and mile.isalpha():
            mileage[i] = 0
        else:
            mileage[i] = int(mile)

    year_normalized = normalize(year)
    price_normalized = normalize(price)
    mileage_normalized = normalize(mileage)

    # one-hot
    model_normalized = np.array(one_hot_encode(model))
    color_normalized = np.array(one_hot_encode(color))
    drivetrain_normalized = np.array(one_hot_encode(drivetrain))
    engine_normalized = np.array(one_hot_encode(engine))
    fuel_type_normalized = np.array(one_hot_encode(fuel_type))
    seller_normalized = np.array(one_hot_encode(seller))

    #
    # for i in range(len(engine)):
    #     print(engine[i], "\t", engine_normalized[i])

    x = np.concatenate(
        (year_normalized, mileage_normalized, model_normalized, color_normalized,
         drivetrain_normalized, engine_normalized, fuel_type_normalized, seller_normalized),
        axis=1)
    y = price_normalized

    np.random.seed(47)
    shuffled_indices = np.random.permutation(len(x))
    x_shuffled = x[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    train_size = int(0.8 * len(x_shuffled))
    val_size = int(0.1 * len(x_shuffled))

    # y_shuffled = np.array([y_shuffled])

    x_train = x_shuffled[:train_size]
    y_train = y_shuffled[:train_size]
    x_val = x_shuffled[train_size:train_size + val_size]
    y_val = y_shuffled[train_size:train_size + val_size]
    x_test = x_shuffled[train_size + val_size:]
    y_test = y_shuffled[train_size + val_size:]

    # how many input neurons I need
    sum_inp_neu = sum([arr.shape[1] for arr in [year_normalized, mileage_normalized, model_normalized,
                                                color_normalized, drivetrain_normalized, engine_normalized,
                                                fuel_type_normalized,
                                                seller_normalized]])
    # REFORMAT Y to 2D
    # print(y_train)
    # yamnt = len(y_train)
    # y_train = np.hstack((y_train, np.zeros((yamnt, 1))))

    # NI, NH, NO, B = sum_inp_neu, sum_inp_neu * 2, 1, 1
    # w1 = np.random.random((NH, NI + B)) * 0.8 - 0.4
    # w2 = np.random.random((NO, NH)) * 0.8 - 0.4
    #
    # # np.set_printoptions(threshold=np.inf)
    #
    # re-format x_input, y_input to proper input sample
    samples = list([
        [x_e, y_e] for x_e, y_e in zip(x_train, y_train)
    ])

    acti, dacti = activation_sigmoid, dactivation_sigmoid
    # acti, dacti = activation_tanh, dactivation_tanh

    # samples = [[[0, 0], [0, 0]], [[0, 1], [0, 1]], [[1, 0], [0, 1]], [[1, 1], [1, 1]]]   # AND ĂŠs OR
    # samples_old = [[[0, 0], [0, 0, 0]], [[0, 1], [0, 1, 1]], [[1, 0], [0, 1, 1]], [[1, 1], [1, 1, 0]]]  # AND, OR ĂŠs XOR

    # print(numpy.shape(samples_old[0][1]))
    # print(numpy.shape(samples[0][1]))
    # print(len(samples[0][0]))

    B = 1
    # nn = [2+B, 4, ADD LAYERS 3, 3]
    nn = [len(samples[0][0]) + B, 4, 5, 3, len(samples[0][1])]
    wl = [numpy.random.random((nn[l + 1], nn[l])) * 0.8 - 0.4 for l in range(len(nn) - 1)]
    # wl = [numpy.random.random((nn[l + 1], nn[l]-1)) * 0.8 - 0.4 for l in range(len(nn) - 1)]
    # wl = [numpy.linspace(-1,1,nn[l]*nn[l+1]).reshape((nn[l+1], nn[l])) for l in range(len(nn)-1)]
    delta = [numpy.zeros((nn[l + 1])) for l in range(len(nn) - 1)]

    epoch = 0
    sumerr = 1.0
    while sumerr >= 0.01 and epoch <= 50:
        sumerr = 0.0
        epoch += 1
        # for inp, out in random.sample(samples, len(samples)):
        for inp, out in samples:
            nl = [numpy.array(list(inp) + [1.0] * B)]
            for l in range(len(nn) - 1):
                # print(wl[l].shape)
                # print(nl[l].shape)
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
