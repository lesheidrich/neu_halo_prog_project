import numpy as np
import pandas as pd
from pprint import pprint as pprint

df = pd.read_csv('../dataset/honda_sell_data.csv')
filtered_cols = df[['Year', 'Model', 'Price', 'Exterior_Color', 'Drivetrain', 'Fuel_Type',
                    'Engine', 'Mileage', 'Seller_Type']]

# cols_to_select = ['Year', 'Model', 'Price', 'Exterior_Color', 'Drivetrain', 'Fuel_Type', 'Engine', 'Mileage', 'Seller_Type']
# selected_columns = df[cols_to_select]
# print(filtered_cols)

# min-max
year = filtered_cols['Year'].to_numpy()
price = filtered_cols['Price'].to_numpy()
mileage = filtered_cols['Mileage'].to_numpy()
# one-hot
model = filtered_cols['Model'].to_numpy()
color = filtered_cols['Exterior_Color'].to_numpy()
drivetrain = filtered_cols['Drivetrain'].to_numpy()
engine = filtered_cols['Engine'].to_numpy()
fuel_type = filtered_cols['Fuel_Type'].to_numpy()
seller = filtered_cols['Seller_Type'].to_numpy()

all_cols = [year, model, price, color, drivetrain, engine, fuel_type, mileage, seller]

del_i = np.where(price == 'Not Priced')[0]
for i, c in enumerate(all_cols):
    all_cols[i] = np.delete(c, del_i, axis=0)
year, model, price, color, drivetrain, engine, fuel_type, mileage, seller = all_cols


# price = np.core.defchararray.replace(price, '$', '').replace(',', '', regex=True)
# price = price.astype(int)
#
# print(price)






#
# #min-max scaling 0-1
# year_normalized = (year - np.min(year)) / (np.max(year) - np.min(year))
# # price_normalized = (price - np.min(price)) / (np.max(price) - np.min(price))
# # mileage_normalized = (mileage - np.min(mileage)) / (np.max(mileage) - np.min(mileage))
#
# #one-hot encoding
seller_labels = list(seller)
lables_inds = [seller_labels.index(label) for label in seller_labels]
tr_model = [[1 if i == l else 0 for i in range(len(set(seller)))] for l in lables_inds]

for i in range(len(seller)):
    print(tr_model[i], "\t", seller[i])
#
# # for i, item in enumerate(mileage):
# #     print(item, "\t", mileage_normalized[i])
#
#
#
#
