import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import logging
logger = logging.getLogger("simple_logger")


def process_data_scaling(data: pd.DataFrame) -> pd.DataFrame:
    """
    imputes missing values, scales features independently, scales between 0 and 1
    Returns:
        scaled dataframe
    """

    data = format_data(data)

    date_time_series = data.pop("Date_Time")  # remove Date_Time column
    date_time_series = date_time_series.apply(lambda x: x[:8])  # get first 8 characters (date only)

    # missing values imputed with np.nan
    imputer = SimpleImputer(missing_values=np.nan)
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    data = data.reset_index(drop=True)

    # feature scaling - ensure they all fall within 0 to 1
    # scales features independently
    # (x - xmin) / (xmax - xmin) -> where xmin is minimum value in feature and xmax is maximum
    scalar = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scalar.fit_transform(data.to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=list(data.columns))

    # add back Date_Time
    df_scaled = df_scaled.astype(float)
    df_scaled.insert(0, "Date_Time", date_time_series)

    return df_scaled


def make_raw_data(data: pd.DataFrame) -> tuple[np.ndarray, pd.Series, list]:
    """removes Date_Time column
    returns raw_data, Date_Time column, column names"""
    date_time_series = data.pop("Date_Time")  # remove Date_Time column
    return data.values, date_time_series, data.columns.tolist()


def format_data(data: pd.DataFrame) -> pd.DataFrame:
    """removes FileId column if present. skips Date_Time column and formats data
    Returns:
        data (pd.DataFrame) with Date_Time column"""

    if "FileId" in data.columns:
        data = data.drop(columns="FileId", axis=1)

    data = data.map(str)

    for col in data.columns[1:]:  # skip first column (Date_Time)
        m_neg = data[col].str.startswith("-")
        data[col] = data[col].str.strip("-")
        data[col] = pd.to_numeric(data[col].astype(str).str.replace(",", "."))  # replace , with . then convert to np
        data.loc[m_neg, col] += -1

    return data


def split_by_test_data_config(data: pd.DataFrame | np.ndarray, config_values
                              ) -> tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame]:
    """returns a split of data, depending on test_data_config type
    data can be a DataFrame or numpy raw data"""

    test_data_config = config_values["test_data_config"]

    logger.info("data split into train_data (for model training) & test_data (for anomaly detection)")

    if isinstance(test_data_config, int):

        # translate line number in csv file to same value in dataframe
        train_length = test_data_config - 2

        logger.info(f"anomaly detection data at file line {test_data_config} , df index {train_length}")

        if train_length < 0:
            raise ValueError(f"split line num of {test_data_config} too short")  # should never happen (load_options.py)

        elif train_length == len(data):
            raise ValueError(f"split line num of {test_data_config} contains no test_data")

        # split data
        original_train_data = data[:train_length]
        original_test_data = data[train_length:]

    elif isinstance(test_data_config, float):
        logger.info(f"training on first {test_data_config * 100}% of data")

        train_length = max(1, int(len(data) * test_data_config))

        # split data
        original_train_data = data[:train_length]
        original_test_data = data[train_length:]

    elif isinstance(test_data_config, type(None)):
        logger.info("anomaly detection on training data")

        original_train_data = data
        original_test_data = original_train_data

    elif isinstance(test_data_config, str):

        test_file_path = test_data_config
        logger.info(f"anomaly detection on data from {test_file_path}")

        test_data = pd.read_csv(test_file_path, sep=";")  # read .csv file and

        common_params = list(set(config_values["parameters"]) & set(test_data.columns))
        common_params.remove("Date_Time")
        common_params.insert(0, "Date_Time")

        original_train_data = data

        original_test_data = process_data_scaling(
            test_data[common_params]  # filter parameters to match original
        )

        if isinstance(original_train_data, np.ndarray):
            original_test_data, _, _ = make_raw_data(original_test_data)

    else:
        raise ValueError("test_data_config not of type int, float, None, or str")

    if (len(original_train_data) == 0) or (len(original_test_data) == 0):
        raise Exception(f"One split is empty. train: {original_train_data.shape}. test: {original_test_data.shape}")

    return original_train_data, original_test_data


def extend_data(data: np.ndarray | pd.DataFrame, steps_in_batch: int) -> np.ndarray | pd.DataFrame:
    """
    extends data with final index, until number datapoints is divisible by steps_in_batch
    """

    # check data can be reshaped by checking number datapoints is divisible by steps_in_batch
    if isinstance(data, np.ndarray):
        while data.shape[0] % steps_in_batch != 0:
            data = np.vstack([data, data[-1]])
    else:
        while data.shape[0] % steps_in_batch != 0:
            data = pd.concat([data, data.tail(1)], ignore_index=True)

    return data


def prepare_DataFrames(data: pd.DataFrame, config_values: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """splits and extends data. Returns 2 dataframes"""

    # add column names to config_values, so dataframes from other .csv files can be filtered.
    config_values["parameters"] = data.columns

    # splits raw_scaled_data depending on test_data_config
    # test_data_config can be str, int, or None. See README.md for more details
    original_train_data, original_test_data = split_by_test_data_config(data, config_values)

    old_length_train = original_train_data.shape[0]
    old_length_test = original_test_data.shape[0]

    logger.info("extending data")
    original_train_data = extend_data(original_train_data, config_values["input_neurons"])
    original_test_data = extend_data(original_test_data, config_values["input_neurons"])

    logger.info(f"train data extended by {original_train_data.shape[0] - old_length_train} datapoints, shape: {original_train_data.shape[0]} datapoints {original_train_data.shape[1]} parameters")
    logger.info(f"test data extended by {original_test_data.shape[0] - old_length_test} datapoints, shape: {original_test_data.shape[0]} datapoints {original_test_data.shape[1]} parameters")

    return original_train_data, original_test_data


if __name__ == "__main__":
    from src.load_options import load_yaml
    import os

    os.chdir("../")
    c = load_yaml("configuration.yml")  # ENV Variables

    a = pd.read_csv(c["train_file_path"], sep=";")
    a = process_data_scaling(a)

    a = a[50:]
    print("test data shape: ", a.shape)

    print("\nextending to be divisible by 12")
    print("extend_data return shape: ", extend_data(a, 12).shape)
    print("")

    oa, ob = prepare_DataFrames(a, c)

    print("\nextended split shapes:")
    print("oa.shape: ", extend_data(oa, 12).shape)
    print("ob.shape: ", extend_data(ob, 12).shape)
    