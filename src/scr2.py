import numpy as np
import math
import pandas as pd


def sigmoid_activation_func(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_activation_func_derivative(y):
    return y * (1.0 - y)


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def one_hot_encode(col: list) -> np.ndarray:
    """
    :param col: list of values to encode
    :return: numpy array of one-hot encoded values
    """
    col_unique = list(set(col))
    col_train = []
    for value in col:
        one_hot = [1 if value == v else 0 for v in col_unique]
        col_train.append(one_hot)
    return np.array(col_train)


def train_neural_network(X, y, hidden_layer_size=10, num_iterations=10000, learning_rate=0.1):
    input_size = X.shape[1]
    output_size = 1

    # Initialize weights and biases
    W1 = np.random.randn(input_size, hidden_layer_size)
    b1 = np.zeros((1, hidden_layer_size))
    W2 = np.random.randn(hidden_layer_size, output_size)
    b2 = np.zeros((1, output_size))

    for i in range(num_iterations):
        # Forward propagation
        Z1 = np.dot(X, W1) + b1
        A1 = sigmoid_activation_func(Z1)
        Z2 = np.dot(A1, W2) + b2
        A2 = sigmoid_activation_func(Z2)

        # Backpropagation
        dZ2 = A2 - y
        dW2 = np.dot(A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * sigmoid_activation_func_derivative(A1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Update weights and biases
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1

    return W1, b1, W2, b2


# Load the normalized data
df = pd.read_csv('../dataset/honda_sell_data.csv')
filtered_cols = df[['Year', 'Model', 'Price', 'Exterior_Color', 'Drivetrain', 'Fuel_Type',
                    'Engine', 'Mileage', 'Seller_Type']]
filtered_cols = filtered_cols[filtered_cols['Price'] != 'Not Priced']

year = normalize_data(filtered_cols['Year'].to_numpy())
price = normalize_data([int(p.replace("$", "").replace(",", "")) for p in filtered_cols['Price'].to_numpy()])
mileage = normalize_data([int(mile) if str(mile).isdigit() else 0 for mile in filtered_cols['Mileage'].to_numpy()])
model = one_hot_encode(filtered_cols['Model'].to_numpy())
color = one_hot_encode(filtered_cols['Exterior_Color'].to_numpy())
drivetrain = one_hot_encode(filtered_cols['Drivetrain'].to_numpy())
engine = one_hot_encode(filtered_cols['Engine'].to_numpy())
fuel_type = one_hot_encode(filtered_cols['Fuel_Type'].to

# Convert normalized data to numpy arrays
year_normalized = np.array(year_normalized)
price_normalized = np.array(price_normalized)
mileage_normalized = np.array(mileage_normalized)
model_normalized = np.array(model_normalized)
color_normalized = np.array(color_normalized)
drivetrain_normalized = np.array(drivetrain_normalized)
engine_normalized = np.array(engine_normalized)
fuel_type_normalized = np.array(fuel_type_normalized)
seller_normalized = np.array(seller_normalized)

# Split data into training, validation, and testing sets
data_size = len(year_normalized)
train_size = int(0.8 * data_size)
val_size = int(0.1 * data_size)
test_size = data_size - train_size - val_size

train_indices = list(range(train_size))
val_indices = list(range(train_size, train_size + val_size))
test_indices = list(range(train_size + val_size, data_size))

train_X = np.column_stack((year_normalized[train_indices], mileage_normalized[train_indices], drivetrain_normalized[train_indices], fuel_type_normalized[train_indices]))
train_y = price_normalized[train_indices]
val_X = np.column_stack((year_normalized[val_indices], mileage_normalized[val_indices], drivetrain_normalized[val_indices], fuel_type_normalized[val_indices]))
val_y = price_normalized[val_indices]
test_X = np.column_stack((year_normalized[test_indices], mileage_normalized[test_indices], drivetrain_normalized[test_indices], fuel_type_normalized[test_indices]))
test_y = price_normalized[test_indices]

# Define sigmoid activation function and its derivative
def sigmoid_activation_func(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_activation_func_derivative(y):
    return y * (1.0 - y)

# Initialize weights and biases
input_size = train_X.shape[1]
hidden_size = 10
output_size = 1

np.random.seed(0)
W1 = 2 * np.random.random((input_size, hidden_size)) - 1
b1 = np.zeros((1, hidden_size))
W2 = 2 * np.random.random((hidden_size, output_size)) - 1
b2 = np.zeros((1, output_size))

# Define forward propagation function
def forward_propagation(X):
    # Input layer to hidden layer
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid_activation_func(z1)

    # makefile
    # Copy code
    # Hidden layer to output layer
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid_activation_func(z2)

    return a1, a2

# Train the neural network
learning_rate = 0.01
epochs = 10000

for epoch in range(epochs):
    # Forward propagation
    a1, a2 = forward_propagation(train_X)

    # makefile
    # Copy code
    # Backward propagation
delta2 = (a2 - train_y) * sigmoid_activation_func_derivative(a2)
dW2 = np.dot(a1.T, delta2)
db2 = np.sum(delta2, axis=0)
delta1 = np.dot(delta2, W2.T) * sigmoid_activation_func_derivative(a1)
dW1 = np.dot(train_X.T, delta1)
db1 = np.sum(delta1, axis=0)

# Update weights and biases
W2 -= learning_rate * dW2
b2 -= learning_rate * db2
W1 -= learning_rate * dW1
b1 -= learning_rate * db1

# Compute loss
loss = np.mean((a2 - train_y) ** 2)

# Print loss for every 1000 epochs
if epoch % 1000 == 0:
    print(f"Epoch: {epoch}, Loss: {loss}")
# Evaluate the model on validation data
val_a1, val_a2 = forward_propagation(val_X)
val_loss = np.mean((val_a2 - val_y) ** 2)
print(f"Validation Loss: {val_loss}")

# Make predictions on test data
test_a1, test_a2 = forward_propagation(test_X)
test_predictions = test_a2 * (max_price - min_price) + min_price
test_actual_prices = test_y * (max_price - min_price) + min_price

# Calculate mean absolute percentage error (MAPE)
mape = np.mean(np.abs(test_actual_prices - test_predictions) / test_actual_prices) * 100
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

# isualize actual prices vs. predicted prices
plt.scatter(test_actual_prices, test_predictions)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs. Predicted Prices')
plt.show()
