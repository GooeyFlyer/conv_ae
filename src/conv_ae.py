import keras
import numpy as np
import tensorflow as tf
import pandas as pd

from src.get_data import process_data_scaling, data_operations
from src.PlottingManager import PlottingManager
from src.AnomalyDetector import AnomalyDetector
from src.LossThresholdCalculator import LossThresholdCalculator


def loss_below_threshold(test_reconstructions: tf.Tensor, test_data: np.ndarray, threshold: float) -> tf.Tensor:
    """
    Returns:
        tensorflow array, of boolean if datapoint loss < threshold, for each datapoint (a.k.a timestamp) in test_data
    """
    loss = keras.losses.mean_absolute_error(y_pred=test_reconstructions, y_true=test_data)
    return tf.math.less(loss, threshold)


def set_draw_reconstructions(draw_reconstructions: str, num_columns: int) -> bool:
    if draw_reconstructions == "yes":
        return True
    elif draw_reconstructions == "no":
        return False
    elif draw_reconstructions == "auto":
        if num_columns <= 20:
            return True
        return False

    print(f"draw_reconstructions is invalid value {draw_reconstructions}.\nDefaulting to False")
    return False


def write_anomalies(test_reconstructions: tf.Tensor, reshaped_test_data: np.ndarray, threshold: float,
                    date_time_series: pd.Series, filter_message: str, file_path: str = "anomaly_stats.txt") -> None:
    """predicts anomalies and saves info to .txt file"""

    # 1 prediction per test_data datapoint
    predictions = loss_below_threshold(test_reconstructions, reshaped_test_data, threshold)
    predictions = tf.reshape(predictions, [-1])  # flatten tensor
    print("\npredictions info: ", tf.size(predictions))

    # saves indices of anomalies (False values) in predictions
    anomaly_indices = [i for i in range(len(predictions)) if not predictions[i]]

    # filters Date_Time Series to just values in test_data
    test_date_time_series = (date_time_series[-len(predictions):]).reset_index()

    # filters test_date_time_series by indices saved in anomalies
    anomalies = test_date_time_series[
        test_date_time_series.index.isin(anomaly_indices)
    ]

    anomalies = anomalies.rename(columns={"index": "file_line_number"})
    anomalies.iloc[:, 0] += 2

    pd.set_option("display.max_rows", None)
    output = f"""Anomalies
{filter_message}
stats:
no. of anomalies in test_data: {len(anomaly_indices)}
percentage of anomalies in test_data: {round(len(anomaly_indices) / len(predictions) * 100, 1)}%
\nTimestamps in test data marked as anomalies:
{anomalies}
"""
    with open(file_path, "w") as file:
        file.write(output)

    print("\nstats saved to anomaly_stats.txt")


def anomaly_detection(data: pd.DataFrame, config_values: dict, filter_message: str):
    """
    normalise data, split data, build model, train model, reconstruct test_data, plot graphs, find anomalies
    """

    raw_scaled_data, date_time_series, channel_names = process_data_scaling(data)
    num_channels = raw_scaled_data.shape[1]  # number of columns / features

    print(num_channels, " parameters\n")

    original_train_data, original_test_data, reshaped_train_data, reshaped_test_data = data_operations(
        raw_scaled_data, config_values["input_neurons"], num_channels, config_values
    )

    verbose = {True: "auto", False: 0}[config_values["verbose_model"]]

    # LossThresholdCalculator initialised here, as it contains error checking for config_values["threshold_quantile"]
    calc = LossThresholdCalculator(config_values["loss"], config_values["threshold_quantile"])

    # build model
    print("building model")
    autoencoder = AnomalyDetector(
        num_input_neurons=config_values["input_neurons"],
        num_features=num_channels,
        strides=config_values["strides"],
        pool_size=config_values["pool_size"],
        kernel_size=config_values["kernel_size"],
        activation=config_values["activation"]
    )
    autoencoder.compile(optimizer=config_values["optimizer"], loss=config_values["loss"], metrics=["accuracy"])

    if config_values["verbose_model"]:
        autoencoder.encoder.summary()
        autoencoder.decoder.summary()

    # train model
    print("training model")
    history = autoencoder.fit(
        reshaped_train_data, reshaped_train_data,
        epochs=config_values["epochs"],
        validation_data=(reshaped_test_data, reshaped_test_data),
        shuffle=False,
        verbose=verbose
    )

    print("\nreconstructing data")
    train_reconstructions = autoencoder.predict(
        reshaped_train_data,
        batch_size=reshaped_train_data.shape[0],
        verbose=verbose
    )  # tf.Tensor
    test_reconstructions = autoencoder.predict(
        reshaped_test_data,
        batch_size=reshaped_test_data.shape[0],
        verbose=verbose
    )  # tf.Tensor

    print("\ncalculating stats of all datapoints")
    train_loss, test_loss, threshold = calc(
        train_reconstructions,
        test_reconstructions,
        reshaped_train_data, reshaped_test_data
    )

    del reshaped_train_data

    print("\nplotting")
    plottingManager = PlottingManager(
        draw_plots=config_values["draw_plots"],  # decides if images are drawn
        draw_reconstructions=set_draw_reconstructions(config_values["draw_reconstructions"], num_channels),
        num_to_show=config_values["num_to_show"],
        error_plot=config_values["error_plot"],
        anomaly_split_len=len(original_test_data)
    )

    plottingManager.plot_reconstructions(
        "train",
        original_train_data,
        train_reconstructions.reshape(1, -1, num_channels)[0],  # flatten array
        loss=train_loss,
        column_names=channel_names
    )
    plottingManager.plot_reconstructions(
        "test",
        original_test_data,
        test_reconstructions.reshape(1, -1, num_channels)[0],  # flatten array
        loss=test_loss,
        column_names=channel_names
    )

    plottingManager.plot_model_loss_val_loss(history)

    del original_train_data, original_test_data, train_reconstructions, history

    plottingManager.plot_loss_histograms(train_loss, test_loss, threshold)

    plottingManager.plot_loss_line_chart("test", test_loss, threshold)
    plottingManager.plot_loss_line_chart("train", train_loss, threshold)

    # plottingManager.plot_zoomed_loss_line_chart("train", train_loss, threshold)
    # plottingManager.plot_zoomed_loss_line_chart("test", test_loss, threshold)

    write_anomalies(test_reconstructions, reshaped_test_data, threshold, date_time_series, filter_message)
