import numpy as np
import pandas as pd

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

year_normalized = (year - np.min(year)) / (np.max(year) - np.min(year))
# price_normalized = (price - np.min(price)) / (np.max(price) - np.min(price))
# mileage_normalized = (mileage - np.min(mileage)) / (np.max(mileage) - np.min(mileage))

print(len(set(model)))
print(set(model))


# for i, item in enumerate(mileage):
#     print(item, "\t", mileage_normalized[i])



"""
Year	0-1	minmax saling
Model		one-hot encoding
Price	0-1	minmax saling
Exterior_Color	one-hot encoding
Drivetrain	one-hot encoding
Engine		encode or preprocess
Mileage	0-1	minmax saling
Seller_Type	one-hot encoding
"""



