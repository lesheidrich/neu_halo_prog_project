import pandas as pd
import numpy as np

from src.data_processor import DataProcessor
#
#
# def normalize(col: list) -> list:
#     """
#     :param col: int based list of column values to normalize
#     :return: normalized (0-1) list reshaped to vertical
#     """
#     arr = (col - np.min(col)) / (np.max(col) - np.min(col))
#     return arr.reshape(-1, 1)
#
#
# def one_hot_encode(col: list) -> list:
#     """
#     :param col: list of values to encode
#     :return: list of one-hot encoded values
#     """
#     col_unique = list(set(col))
#     col_train = []
#     for value in col:
#         one_hot = [1 if value == v else 0 for v in col_unique]
#         col_train.append(one_hot)
#     return col_train
#
#
# df = pd.read_csv('../dataset/honda_sell_data_OG.csv')
# filtered_cols = df[['Year', 'Model', 'Price', 'Exterior_Color', 'Drivetrain', 'Fuel_Type',
#                     'Engine', 'Mileage', 'Seller_Type']]
# filtered_cols = filtered_cols[filtered_cols['Price'] != 'Not Priced']
#
# year = filtered_cols['Year'].to_numpy()
# price = filtered_cols['Price'].to_numpy()
# mileage = filtered_cols['Mileage'].to_numpy()
# model = filtered_cols['Model'].to_numpy()  # lehet le kell csokkenteni
# color = filtered_cols['Exterior_Color'].to_numpy()
# drivetrain = filtered_cols['Drivetrain'].to_numpy()
# engine = filtered_cols['Engine'].to_numpy()
# fuel_type = filtered_cols['Fuel_Type'].to_numpy()
# seller = filtered_cols['Seller_Type'].to_numpy()
#
# # min-max
# price = [int(p.replace("$", "").replace(",", "")) for p in price]
# for i, mile in enumerate(mileage):
#     if isinstance(mile, float) or mile == '–':
#         mileage[i] = 0
#     elif isinstance(mile, str) and mile.isalpha():
#         mileage[i] = 0
#     else:
#         mileage[i] = int(mile)
#
# year_normalized = normalize(year)
# price_normalized = normalize(price)
# mileage_normalized = normalize(mileage)
#
#
#
# # one-hot
# model_normalized = np.array(one_hot_encode(model))
# color_normalized = np.array(one_hot_encode(color))
# drivetrain_normalized = np.array(one_hot_encode(drivetrain))
# engine_normalized = np.array(one_hot_encode(engine))
# fuel_type_normalized = np.array(one_hot_encode(fuel_type))
# seller_normalized = np.array(one_hot_encode(seller))
#
# #
# # for i in range(len(engine)):
# #     print(engine[i], "\t", engine_normalized[i])
#
# x = np.concatenate(
#     (year_normalized, mileage_normalized, model_normalized, color_normalized,
#      drivetrain_normalized, engine_normalized, fuel_type_normalized, seller_normalized),
#     axis=1)
# y = price_normalized
#
# np.random.seed(47)
# shuffled_indices = np.random.permutation(len(x))
# x_shuffled = x[shuffled_indices]
# y_shuffled = y[shuffled_indices]
#
# train_size = int(0.8 * len(x_shuffled))
# val_size = int(0.1 * len(x_shuffled))
#
# x_train = x_shuffled[:train_size]
# y_train = y_shuffled[:train_size]
# x_val = x_shuffled[train_size:train_size + val_size]
# y_val = y_shuffled[train_size:train_size + val_size]
# x_test = x_shuffled[train_size + val_size:]
# y_test = y_shuffled[train_size + val_size:]
# # model_test
# model_resh = model.reshape(-1, 1)
# model_shuf = model_resh[shuffled_indices]
# model_y = model_shuf[train_size + val_size:]
#
# # how many input neurons I need
# sum_inp_neu = sum([arr.shape[1] for arr in [year_normalized, mileage_normalized, model_normalized,
#                                             color_normalized, drivetrain_normalized, engine_normalized,
#                                             fuel_type_normalized, seller_normalized]])
#
# np.set_printoptions(threshold=np.inf)
#
# # re-format x_input, y_input to proper input sample
# samples = list([
#     [x_e, y_e] for x_e, y_e in zip(x_train, y_train)
# ])
#
# test_samples = list([
#     [x_t, y_t] for x_t, y_t in zip(x_test, y_test)
# ])

dp = DataProcessor("honda_sell_data_MODEL.csv")
dp.pre_process()

