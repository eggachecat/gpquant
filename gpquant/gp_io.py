import pandas as pd


def read_data(data_path):
    df = pd.read_csv(data_path)
    x_data = df.drop("price", axis=1)
    price = df["price"]

    return x_data.as_matrix(), price.as_matrix()
