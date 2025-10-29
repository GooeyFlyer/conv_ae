import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf


def process_data_tensor_flow(file_path: str):
    """splits data, normalises data
    issue as it does not normalise each column separately
    assumes all columns are related"""

    data = format_data(pd.read_csv(file_path, sep=";"))
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
    imputes missing values, scales columns independently

    Returns:
        numpy.ndarray: raw_scaled_data
        pandas.Series: Series of time stamps
        clist: array of column names
    """
    data = format_data(pd.read_csv(file_path, sep=";"))

    date_time_series = data.pop("Date_Time")  # remove Date_Time column
    date_time_series = date_time_series.apply(lambda x: x[:-13])  # remove last 13 characters (time info)

    column_names = data.columns.tolist()

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

    return raw_scaled_data, date_time_series, column_names


def split_data(raw_scaled_data: np.ndarray, len_column_names):
    # split data
    train_data, test_data = train_test_split(raw_scaled_data, test_size=0.2, shuffle=False)

    print("train_data size: ", len(train_data))
    print("test_data size: ", len(test_data))

    if len(test_data[0]) != len_column_names:
        raise ValueError("len(test_data[0]) not equal to len_column_names")

    return train_data, test_data


def format_data(data: pd.DataFrame) -> pd.DataFrame:
    """skips Date_Time column and formats data
    Returns:
        data (pd.DataFrame)"""

    # data = data.drop(columns="Date_Time", axis=1)
    if "FileId" in data.columns:
        data = data.drop(columns="FileId", axis=1)

    data = data.map(str)

    for col in data.columns[1:]:
        m_neg = data[col].str.startswith("-")
        data[col] = data[col].str.strip("-")
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", "."))  # replace , with . then convert to np
        data.loc[m_neg, col] += -1

    print("dataframe shape: ", data.shape)

    return data


def split_at_index(target_index: int, raw_scaled_data: np.ndarray, common_factor: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the index where the length of both the left and right split are divisible by common_factor.
    Attempts index target_index first, then decreases index.
    """

    # tries all index values from target_index decreasing to 1
    for train_length in range(target_index, 1, -1):
        test_length = raw_scaled_data.shape[0] - train_length  # test split

        # both are divisible by common_factor
        if train_length % common_factor == 0 and test_length % common_factor == 0:

            # split data
            original_train_data = raw_scaled_data[:train_length]
            original_test_data = raw_scaled_data[train_length:]

            print(f"data split into train data & anomaly detection data, at index {train_length}")
            return original_train_data, original_test_data

    gcd = math.gcd(target_index, raw_scaled_data.shape[0] - target_index)

    # error if split not found where both are divisible by common_factor
    raise ValueError(f"""\n\n
Split cannot be found:
    split_index: {target_index}
    raw_scaled_data.shape: {raw_scaled_data.shape}
    common_factor: {common_factor}

Greatest Common Denominator of {target_index} and {raw_scaled_data.shape[0] - target_index} = {gcd}
""")


def parse_test_data_config(test_data_config, raw_scaled_data) -> tuple[np.ndarray, np.ndarray]:
    """returns a split of raw_scaled_data, depending on test_data_config type"""

    if isinstance(test_data_config, int):

        train_length = test_data_config
        # split data
        original_train_data = raw_scaled_data[:train_length]
        original_test_data = raw_scaled_data[train_length:]
        print(f"data split into train data & anomaly detection data, at index {train_length}")

    elif isinstance(test_data_config, type(None)):

        original_train_data = raw_scaled_data
        original_test_data = original_train_data
        print("anomaly detection on training data")

    elif isinstance(test_data_config, str):

        test_file_path = test_data_config
        original_train_data = raw_scaled_data
        original_test_data, _, _ = process_data_scaling(test_file_path)
        print(f"anomaly detection on data from {test_file_path}")

    else:
        raise ValueError("test_data_config not of type int, None, or str")

    return original_train_data, original_test_data


def extend_data(data: np.ndarray, steps_in_batch: int) -> np.ndarray:
    """
    extends data with final index, until number datapoints is divisible by steps_in_batch
    """

    # check data can be reshaped by checking number datapoints is divisible by steps_in_batch
    while data.shape[0] % steps_in_batch != 0:
        data = np.vstack([data, data[-1]])

    return data


if __name__ == "__main__":

    # print(get_data(10, 3, 2))

    a, _, _ = process_data_scaling("../data/FeatureDataSel.csv")

    a = a[50:]
    print("test data shape: ", a.shape)

    print("\nextending to be divisible by 12")
    print("extend_data return shape: ", extend_data(a, 12).shape)
    print("")

    oa, ob = parse_test_data_config(test_data_config=1200, raw_scaled_data=a)

    print("\nextended split shapes:")
    print("oa.shape: ", extend_data(oa, 12).shape)
    print("ob.shape: ", extend_data(ob, 12).shape)
