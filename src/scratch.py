
labels = ["sport", "news", "weather", "sport", "weather"]
lables_inds = [labels.index(label) for label in labels]
Train_y = [[1 if i == l else 0 for i in range(3)] for l in lables_inds]

print(lables_inds)
print(Train_y)


# import numpy as np
# import pandas as pd
#
# # Example dataframe
# df = pd.DataFrame({'Model': ['Model A', 'Model B', 'Model C', 'Model A', 'Model B']})
#
# # Convert 'Model' column to NumPy array
# model = df['Model'].to_numpy()
#
# # Get unique values in 'model' array
# unique_values = np.unique(model)
#
# # Create an empty 2D array to store the one-hot encoded values
# model_encoded = np.zeros((len(model), len(unique_values)))
#
# # Iterate through 'model' array and set corresponding one-hot encoded value to 1
# for i, value in enumerate(model):
#     j = np.where(unique_values == value)[0][0]
#     model_encoded[i, j] = 1
#
# # Print the encoded array
# print(model_encoded)
