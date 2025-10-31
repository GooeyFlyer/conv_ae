import keras
import numpy as np
import tensorflow as tf
import pandas as pd

from src.get_data import process_data_scaling, parse_test_data_config, extend_data
from src.PlottingManager import PlottingManager
from src.load_options import load_yaml
from src.AnomalyDetector import AnomalyDetector


def predict(test_reconstructions: tf.Tensor, test_data: np.ndarray, threshold: float) -> tf.Tensor:
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


def predict_anomalies(test_reconstructions: tf.Tensor, reshaped_test_data: np.ndarray, threshold: float,
                      date_time_series: pd.Series, file_path: str = "anomaly_stats.txt") -> None:
    """predicts anomalies and saves info to .txt file"""

    predictions = predict(test_reconstructions, reshaped_test_data, threshold)  # 1 prediction per test_data datapoint
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
    output = f"""stats:
no. of anomalies in test_data: {len(anomaly_indices)}
percentage of anomalies in test_data: {round(len(anomaly_indices)/len(predictions)*100, 1)}%
\nTimestamps in test data marked as anomalies:
{anomalies}
"""
    with open(file_path, "w") as file:
        file.write(output)

    print("\nstats saved to anomaly_stats.txt")


def calculate_loss_and_threshold(train_reconstructions: tf.Tensor, test_reconstructions: tf.Tensor,
                                 reshaped_train_data: np.ndarray, reshaped_test_data: np.ndarray):
    """threshold is calculated from reshaped_train_data"""

    print("\ncalculating test loss, threshold, & train loss")

    train_loss = tf.reshape(
        keras.losses.mean_absolute_error(y_pred=train_reconstructions, y_true=reshaped_train_data),
        shape=[-1]
    )

    # choose threshold that is one standard deviation above the mean
    threshold = float(np.mean(train_loss) + np.std(train_loss))
    print("calculated anomaly Threshold: ", threshold)

    test_loss = tf.reshape(
        keras.losses.mean_absolute_error(y_pred=test_reconstructions, y_true=reshaped_test_data),
        shape=[-1]
    )

    return train_loss, test_loss, threshold


def conv_ae():
    """
    normalise data, split data, build model, train model, reconstruct test_data, plot graphs, find anomalies
    """

    config_values = load_yaml("configuration.yml")  # ENV Variables

    # normalise data
    raw_scaled_data, date_time_series, channel_names = process_data_scaling(config_values["train_file_path"])

    steps_in_batch = 16  # no. of neurons
    epochs = 200

    # splits raw_scaled_data depending on test_data_config
    # test_data_config can be str, int, or None. See README.md for more details
    original_train_data, original_test_data = parse_test_data_config(config_values["test_data_config"], raw_scaled_data)
    train_len = original_train_data.shape[0]
    test_len = original_test_data.shape[0]

    num_channels = raw_scaled_data.shape[1]  # number channels

    del raw_scaled_data

    print("extending data")
    original_train_data = extend_data(original_train_data, steps_in_batch)
    original_test_data = extend_data(original_test_data, steps_in_batch)

    print(f"train_data extended by {original_train_data.shape[0] - train_len} datapoints")
    print(f"test_data extended by {original_test_data.shape[0] - test_len} datapoints")

    print("\ntrain_data.shape: ", original_train_data.shape)
    print("test_data.shape: ", original_test_data.shape)

    # batch shape, steps_in_batch, num features
    reshaped_train_data = original_train_data.reshape(-1, steps_in_batch, num_channels)
    reshaped_test_data = original_test_data.reshape(-1, steps_in_batch, num_channels)

    del original_train_data

    print("\n")

    # build model
    print("building model")
    autoencoder = AnomalyDetector(steps_in_batch, num_channels)
    autoencoder.compile(optimizer="adam", loss="mae", metrics=["accuracy"])

    if config_values["verbose_model"]:
        autoencoder.encoder.summary()
        autoencoder.decoder.summary()

    # train model
    print("training model")
    history = autoencoder.fit(
        reshaped_train_data, reshaped_train_data,
        epochs=epochs,
        validation_data=(reshaped_test_data, reshaped_test_data),
        shuffle=False,
        verbose={True: "auto", False: 0}[config_values["verbose_model"]],
    )

    print("\nreconstructing data")
    train_reconstructions = autoencoder.predict(
        reshaped_train_data,
        batch_size=reshaped_train_data.shape[0],
        verbose={True: "auto", False: 0}[config_values["verbose_model"]]
    )
    test_reconstructions = autoencoder.predict(
        reshaped_test_data,
        batch_size=reshaped_test_data.shape[0],
        verbose={True: "auto", False: 0}[config_values["verbose_model"]]
    )

    print("\ncalculating stats of all datapoints")
    train_loss, test_loss, threshold = calculate_loss_and_threshold(
        train_reconstructions,
        test_reconstructions,
        reshaped_train_data, reshaped_test_data
    )

    del train_reconstructions, reshaped_train_data

    print("\nplotting")
    plottingManager = PlottingManager(
        draw_plots=config_values["draw_plots"],  # decides if images are drawn
        draw_reconstructions=set_draw_reconstructions(config_values["draw_reconstructions"], num_channels),
        num_to_show=config_values["num_to_show"],
        anomaly_split_len=len(original_test_data),
    )

    # flatten test_reconstructions
    plottingManager.plot_reconstructions(
        original_test_data, test_reconstructions.reshape(1, -1, num_channels)[0], channel_names
    )

    del original_test_data

    plottingManager.plot_model_loss_val_loss(history)

    plottingManager.plot_loss_histograms(train_loss, test_loss, threshold)
    plottingManager.plot_loss_bar_chart(test_loss, threshold)

    predict_anomalies(test_reconstructions, reshaped_test_data, threshold, date_time_series)


if __name__ == "__main__":
    conv_ae()
