# import numpy as np
#
# # Assume you have already preprocessed your data and have X (features) and y (target) as numpy arrays
#
# # Shuffle the data randomly with a fixed random seed for reproducibility
# np.random.seed(42)
# shuffled_indices = np.random.permutation(len(x))
# x_shuffled = x[shuffled_indices]
# y_shuffled = y[shuffled_indices]
#
# # Calculate the sizes of training, validation, and testing sets
# train_size = int(0.8 * len(x_shuffled))
# val_size = int(0.1 * len(x_shuffled))
#
# # Split the data into training, validation, and testing sets
# x_train = x_shuffled[:train_size]
# y_train = y_shuffled[:train_size]
#
# x_val = x_shuffled[train_size:train_size + val_size]
# y_val = y_shuffled[train_size:train_size + val_size]
#
# x_test = x_shuffled[train_size + val_size:]
# y_test = y_shuffled[train_size + val_size:]
