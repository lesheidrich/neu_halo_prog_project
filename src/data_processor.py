import numpy as np
import pandas as pd


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


class DataProcessor:
    def __init__(self, table: str):
        self.df = pd.read_csv(f'../dataset/{table}')
        self.filtered_cols = self.df[['Year', 'Model', 'Price', 'Exterior_Color', 'Drivetrain', 'Fuel_Type',
                                      'Engine', 'Mileage', 'Seller_Type']]
        self.test_samples = []
        self.train_samples = []
        self.sum_inp_neu = None
        self.model_label_y = None

    def pre_process(self):
        self.filtered_cols = self.filtered_cols[self.filtered_cols['Price'] != 'Not Priced']

        year = self.filtered_cols['Year'].to_numpy()
        price = self.filtered_cols['Price'].to_numpy()
        mileage = self.filtered_cols['Mileage'].to_numpy()
        model = self.filtered_cols['Model'].to_numpy()
        color = self.filtered_cols['Exterior_Color'].to_numpy()
        drivetrain = self.filtered_cols['Drivetrain'].to_numpy()
        engine = self.filtered_cols['Engine'].to_numpy()
        fuel_type = self.filtered_cols['Fuel_Type'].to_numpy()
        seller = self.filtered_cols['Seller_Type'].to_numpy()

        print(len(model))

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
        val_size = int(0.2 * len(x_shuffled))

        x_train = x_shuffled[:train_size]
        y_train = y_shuffled[:train_size]
        x_test = x_shuffled[train_size:]
        y_test = y_shuffled[train_size:]

        # model_test
        model_resh = model.reshape(-1, 1)
        model_shuf = model_resh[shuffled_indices]
        self.model_label_y = model_shuf[train_size + val_size:]

        # input neuron count
        self.sum_inp_neu = sum([arr.shape[1] for arr in [year_normalized, mileage_normalized, model_normalized,
                                                         color_normalized, drivetrain_normalized, engine_normalized,
                                                         fuel_type_normalized, seller_normalized]])

        # re-format x_input, y_input to proper input sample
        self.train_samples = list([
            [x_e, y_e] for x_e, y_e in zip(x_train, y_train)
        ])

        self.test_samples = list([
            [x_t, y_t] for x_t, y_t in zip(x_test, y_test)
        ])
