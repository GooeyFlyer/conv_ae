import random
import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf


def process_data_tensor_flow(file_path: str):
    data = preprocess_data(pd.read_csv(file_path, sep=";"))
    data = data.iloc[:, :-1]
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


def singleStepSampler(df: pd.DataFrame, window: int) -> (np.ndarray, np.ndarray):
    """Prepares the data for single-step time-series forecasting"""
    xRes = []  # store input features
    yRes = []  # store target values

    # create sequences of input features and corresponding target values based on window size
    # input features constructed as a sequence of windowed data points,
    # where each data point is a list containing values from each column of the dataframe
    for i in range(0, len(df) - window):
        res = []
        for j in range(0, window):
            r = []
            for col in df.columns:
                r.append(df[col][i + j])
            res.append(r)
        xRes.append(res)
        # filter output columns here if needed
        yRes.append(df.iloc[i + window].values)  # filter columns here
    return np.array(xRes), np.array(yRes)


def process_data_scaling(file_path: str):
    """
    TODO:
        fix this
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

    # apply singleStepSample with window size of 20
    (xVal, yVal) = singleStepSampler(df_scaled, window=20)

    # a constant split with a value of 0.85 is defined, specifying proportion of data to be used for training
    # xVal and yVal are split into training and testing sets according to the split ratio.
    # training set has 0.85% of data, testing has 0.15%
    SPLIT = 0.85

    x_train = xVal[:int(SPLIT * len(xVal))]
    y_train = yVal[:int(SPLIT * len(yVal))]
    x_test = xVal[int(SPLIT * len(xVal)):]
    y_test = yVal[int(SPLIT * len(yVal)):]


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
