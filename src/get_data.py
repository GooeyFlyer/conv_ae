import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


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


def parse_test_data_config(test_data_config, raw_scaled_data) -> tuple[np.ndarray, np.ndarray]:
    """returns a split of raw_scaled_data, depending on test_data_config type"""

    if isinstance(test_data_config, int):

        # translate line number in csv file to same value in dataframe
        train_length = test_data_config - 2
        if train_length < 0:
            raise ValueError(f"split line num of {test_data_config} too short")  # should never happen (load_options.py)

        elif train_length == len(raw_scaled_data):
            raise ValueError(f"split line num of {test_data_config} contains no test_data")

        # split data
        original_train_data = raw_scaled_data[:train_length]
        original_test_data = raw_scaled_data[train_length:]
        print(
            "data split into train & anomaly detection data,",
            f"at file line {test_data_config} , df index {train_length}"
        )

    elif isinstance(test_data_config, type(None)):

        original_train_data = raw_scaled_data
        original_test_data = original_train_data
        print("anomaly detection on training data")

    elif isinstance(test_data_config, str):

        test_file_path = test_data_config
        original_train_data = raw_scaled_data
        original_test_data, _, _ = process_data_scaling(test_file_path)

        if original_train_data.shape[1] != original_test_data.shape[1]:
            raise ValueError("train data and anomaly detection data must have the same number of channels")

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

    from src.load_options import load_yaml
    import os

    os.chdir("../")
    config_values = load_yaml("configuration.yml")  # ENV Variables

    a, _, _ = process_data_scaling(config_values["train_file_path"])

    a = a[50:]
    print("test data shape: ", a.shape)

    print("\nextending to be divisible by 12")
    print("extend_data return shape: ", extend_data(a, 12).shape)
    print("")

    oa, ob = parse_test_data_config(test_data_config=1200, raw_scaled_data=a)

    print("\nextended split shapes:")
    print("oa.shape: ", extend_data(oa, 12).shape)
    print("ob.shape: ", extend_data(ob, 12).shape)
