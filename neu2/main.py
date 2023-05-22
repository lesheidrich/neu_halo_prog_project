import numpy as np
import pandas as pd

from neu2 import losses
from neu2.activation import Activation
from neu2.dense import Dense
from neu2.neural_network import NeuralNetwork


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
    df = pd.read_csv('../dataset/honda_sell_data_OG.csv')
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

    # print(y_train)
    # yamnt = len(y_train)
    #
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

    # acti, dacti = activation_tanh, dactivation_tanh

    # samples = [[[0, 0], [0, 0]], [[0, 1], [0, 1]], [[1, 0], [0, 1]], [[1, 1], [1, 1]]]   # AND ĂŠs OR
    # samples = [[[0, 0], [0, 0, 0]], [[0, 1], [0, 1, 1]], [[1, 0], [0, 1, 1]], [[1, 1], [1, 1, 0]]]  # AND, OR ĂŠs XOR

    print("full", samples[0])
    print("x", x_train.shape)
    print("x", y_train.shape)

    # for cnt in range(100):
    #     sumerr = 0.0
    #     for inp, outpt in samples:
    #         x = np.array(list(inp) + [1.0] * B)
    #
    #         h = sigmoid_activation_func(np.dot(w1, x))
    #         y = sigmoid_activation_func(np.dot(w2, h))
    #         error = outpt - y
    #         deltao = error * sigmoid_activation_func_derivative(y)
    #         deltah = np.dot(deltao, w2) * sigmoid_activation_func_derivative(h)
    #         w2 += 0.5 * deltao.reshape((NO, 1)) * h.reshape((1, NH))
    #         w1 += 0.5 * deltah.reshape((NH, 1)) * x.reshape((1, NI + B))
    #         sumerr += sum(error ** 2)
    #     if sumerr < 0.01:
    #         break
    # print(cnt, sumerr)

    # print(np.asmatrix(y_train))

    network = [
        Dense(sum_inp_neu, 5),
        Activation(),
        Dense(5, 50),
        Activation(),
        Dense(50, 2),
        Activation()
    ]

    epochs = 50
    learrate = 0.5
    NN = NeuralNetwork(network, losses.mse, losses.mse_prime)
    NN.train(np.asmatrix(x_train), np.asmatrix(y_train).transpose(), epochs, learrate)
