import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf


def process_data_tensor_flow(file_path: str):
    """splits data, normalises data
    issue as it does not normalise each column separately
    assumes all columns are related"""

    data = preprocess_data(pd.read_csv(file_path, sep=";"))
    raw_data = data.values  # numpy array

    # split data
    train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=21)  # test_size is fraction of data

    # normalise data to [0, 1] - issue as it does not normalise each column separately
    min_val = tf.reduce_min(train_data)
    max_val = tf.reduce_max(train_data)

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data = tf.cast(train_data, tf.float32)
    test_data = tf.cast(test_data, tf.float32)

    return train_data, test_data


def process_data_scaling(file_path: str):
    """
    imputes missing values, scales columns independently, splits data
    """
    data = preprocess_data(pd.read_csv(file_path, sep=";"))

    # missing values imputed with np.nan
    imputer = SimpleImputer(missing_values=np.nan)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data = data.reset_index(drop=True)

    # feature scaling - ensure they all fall within 0 to 1
    scalar = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scalar.fit_transform(data.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=list(data.columns))

    # use for target scaling of specific columns
    # target_scalar = MinMaxScaler(feature_range=(0, 1))

    df_scaled = df_scaled.astype(float)
    raw_scaled_data = df_scaled.values

    # split data
    train_data, test_data = train_test_split(raw_scaled_data, test_size=0.2, random_state=21)

    return train_data, test_data


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Drops Date_Time column and formats data
    Returns:
        data (pd.DataFrame)"""
    data = data.drop(columns="Date_Time", axis=1)

    for col in data.columns:
        m_neg = data[col].str.startswith("-")
        data[col] = data[col].str.strip("-")
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", "."))
        data.loc[m_neg, col] += -1

    print(data.shape)

    return data


if __name__ == "__main__":

    # print(get_data(10, 3, 2))

    print(preprocess_data(pd.read_csv("FeatureDataSel.csv")).head())
