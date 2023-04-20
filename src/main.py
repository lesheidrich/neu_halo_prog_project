import numpy as np
import pandas as pd
from pprint import pprint as pprint

df = pd.read_csv('../dataset/honda_sell_data.csv')

cols_to_select = ['Year', 'Model', 'Price', 'Exterior_Color', 'Drivetrain', 'Engine', 'Mileage', 'Seller_Type']
selected_columns = df[cols_to_select]

year = selected_columns['Year'].to_numpy()
price = selected_columns['Price'].to_numpy()
mileage = selected_columns['Mileage'].to_numpy()

model = selected_columns['Model'].to_numpy()
color = selected_columns['Exterior_Color'].to_numpy()
drivetrain = selected_columns['Drivetrain'].to_numpy()
engine = selected_columns['Engine'].to_numpy()
seller = selected_columns['Seller_Type'].to_numpy()

all_cols = [year, model, price, color, drivetrain, engine, mileage, seller]
for col in all_cols:
    print(type(col[0]), col[0])

#TODO: convert mileage to int, try catch, if doesn't work might be shitty char, then 0 or del
# mileage = mileage.astype(int)
# print(mileage)

#min-max scaling 0-1
year_normalized = (year - np.min(year)) / (np.max(year) - np.min(year))
# price_normalized = (price - np.min(price)) / (np.max(price) - np.min(price))
# mileage_normalized = (mileage - np.min(mileage)) / (np.max(mileage) - np.min(mileage))

#one-hot encoding
model_labels = list(set(model))
lables_inds = [model_labels.index(label) for label in model_labels]
tr_model = [[1 if i == l else 0 for i in range(len(set(model)))] for l in lables_inds]

print(tr_model)

# for i, item in enumerate(mileage):
#     print(item, "\t", mileage_normalized[i])





