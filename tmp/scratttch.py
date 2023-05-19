import pandas as pd
import numpy as np

from src.data_processor import DataProcessor


dp = DataProcessor("honda_sell_data_MODEL.csv")
dp.pre_process()

samples = dp.train_samples
test_samples = dp.test_samples



def activation_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x.astype(float)))  # (0..1)


def dactivation_sigmoid(x):
    return x * (1.0 - x)


"""
    multiple hidden layers
"""

# TRAINING
acti, dacti = activation_sigmoid, dactivation_sigmoid
# acti, dacti = activation_tanh, dactivation_tanh


B = 1
nn = [dp.sum_inp_neu + B, 85, 50, 15, len(samples[0][1])]
wl = [np.random.random((nn[l + 1], nn[l])) * 0.8 - 0.4 for l in range(len(nn) - 1)]
delta = [np.zeros((nn[l + 1])) for l in range(len(nn) - 1)]

epoch = 0
sumerr = 1.0
while sumerr >= 0.01 and epoch <= 5:
    sumerr = 0.0
    epoch += 1
    for inp, out in samples:
        nl = [np.array(list(inp) + [1.0] * B)]
        for l in range(len(nn) - 1):
            nl.append(acti(np.dot(wl[l], nl[l])))
        error = out - nl[-1]
        for l in reversed(range(len(nn) - 1)):
            if l == len(nn) - 2:
                delta[l][:] = error * dacti(nl[-1])
            else:
                np.dot(delta[l + 1], wl[l + 1], out=delta[l])
                delta[l] *= dacti(nl[l + 1])

            wl[l] += 0.5 * delta[l].reshape((nn[l + 1], 1)) * nl[l].reshape((1, nn[l]))

        sumerr += sum(error ** 2)
print(epoch, sumerr)

# TESTING
test_results = []
for inp, out in test_samples:
    nl = [np.array(list(inp) + [1.0] * B)]
    for l in range(len(nn) - 1):
        nl.append(acti(np.dot(wl[l], nl[l])))

    test_results.append(nl[-1])

# for y in range(len(dp.model_label_y)):
#     print("Model:", dp.model_label_y[y], "\tNeu:", test_results[y], "\tAns:", dp.y_test[y], "\tDiff:",
#           abs(test_results[y] - dp.y_test[y]))


df = pd.DataFrame(columns=["Model", "Neu", "Ans", "Diff"])

for y in range(len(dp.model_label_y)):
    model = dp.model_label_y[y]
    neu = test_results[y]
    ans = dp.y_test[y]
    diff = (abs(test_results[y] - dp.y_test[y]) * 100).round(4)
    row = pd.DataFrame({"Model": [model], "Neu": [neu], "Ans": [ans], "Diff": [diff]})
    df = pd.concat([df, row], ignore_index=True)

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
print(df[["Model", "Neu", "Ans", "Diff"]].to_string(index=False))


