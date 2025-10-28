import keras
import numpy as np
import tensorflow as tf
import pandas as pd
import math

from src.get_data import process_data_scaling
from src.PlottingManager import PlottingManager
from src.load_options import load_yaml
from src.AnomalyDetector import AnomalyDetector

# TODO: Run trained model on combined train_data and test_data


def predict(model: keras.Model, test_data: np.ndarray, threshold: float) -> tf.Tensor:
    """
    Returns:
        tensorflow array, of boolean if datapoint loss < threshold, for each datapoint (a.k.a timestamp) in test_data
    """
    reconstructions = model(test_data)  # test_data ran through trained model. returns tensor with same shape as input
    loss = keras.losses.mean_absolute_error(y_pred=reconstructions, y_true=test_data)
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


def predict_anomalies(autoencoder: keras.Model, reshaped_test_data: np.ndarray, threshold: float,
                      date_time_series: pd.Series, file_path: str = "anomaly_stats.txt") -> None:
    """predicts anomalies and saves info to .txt file"""

    predictions = predict(autoencoder, reshaped_test_data, threshold)  # 1 prediction per test_data datapoint
    predictions = tf.reshape(predictions, [-1])  # flatten tensor
    print("predictions info: ", tf.size(predictions))
    print(predictions[0])

    # saves indices of anomalies (False values) in predictions
    anomaly_indices = [i for i in range(len(predictions)) if not predictions[i]]

    # filters Date_Time Series to just values in test_data
    test_date_time_series = (date_time_series[-len(predictions):]).reset_index()

    # filters test_date_time_series by indices saved in anomalies
    anomalies = test_date_time_series[
        test_date_time_series.index.isin(anomaly_indices)
    ]

    pd.set_option("display.max_rows", None)
    output = f"""stats:
no. of anomalies in test_data: {len(anomaly_indices)}
percentage of anomalies in test_data: {round(len(anomaly_indices)/len(predictions)*100, 1)}%
\nTimestamps in test data marked as anomalies:
test_data index, df index, Date_Time
{anomalies}
"""
    with open(file_path, "w") as file:
        file.write(output)

    print("\nstats saved to anomaly_stats.txt")


def calculate_loss_and_threshold(autoencoder: keras.Model,
                                 reshaped_train_data: np.ndarray, reshaped_test_data: np.ndarray):
    """threshold is calculated from reshaped_test_data"""
    # reconstruction error for training data
    print("calculating test loss, threshold, & train loss")
    reconstructions = autoencoder.predict(reshaped_train_data)  # reconstructs training data (contains anomalies)
    train_loss = tf.reshape(keras.losses.mean_absolute_error(y_pred=reconstructions, y_true=reshaped_train_data), [-1])

    # choose threshold that is one standard deviation above the mean
    threshold = float(np.mean(train_loss) + np.std(train_loss))
    print("calculated anomaly Threshold: ", threshold)

    # reconstruction error for test data
    reconstructions = autoencoder.predict(reshaped_test_data)  # reconstructs testing data (contains anomalies)
    test_loss = tf.reshape(keras.losses.mean_absolute_error(y_pred=reconstructions, y_true=reshaped_test_data), [-1])

    return train_loss, test_loss, threshold


def conv_ae():
    """
    split data, normalise data, build model, train model, reconstruct test_data, plot graphs, find anomalies
    """

    # ENV Variables
    config_values = load_yaml("configuration.yml")

    file_path = config_values["file_path"]  # file_path of .csv file
    draw_plots = config_values["draw_plots"]  # decides if images are drawn
    draw_reconstructions = config_values["draw_reconstructions"]  # decides if reconstruction plots are drawn
    num_to_show = config_values["num_to_show"]  # datapoints from index 0 (inclusive) that are plotted

    # split & normalise data
    original_train_data, original_test_data, date_time_series, column_names = process_data_scaling(file_path)

    steps_in_batch = math.gcd(original_train_data.shape[0], original_test_data.shape[0])  # highest common factor
    steps_in_batch = 4  # no. of neurons

    num_columns = original_train_data.shape[1]  # number channels

    # batch shape, steps_in_batch, channels for datapoint
    reshaped_train_data = original_train_data.reshape(-1, steps_in_batch, num_columns)
    reshaped_test_data = original_test_data.reshape(-1, steps_in_batch, num_columns)

    print(reshaped_train_data.shape)
    print(reshaped_test_data.shape)

    # build model
    print("building model")
    autoencoder = AnomalyDetector(steps_in_batch, num_columns)
    autoencoder.compile(optimizer="adam", loss="mae")
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    # train model
    print("training model")
    history = autoencoder.fit(
        reshaped_train_data, reshaped_train_data,
        epochs=20,
        validation_data=(reshaped_test_data, reshaped_test_data),
        shuffle=True
    )

    # reconstructing test_data
    # TODO: check if this is the same as AnomalyDetector.call(reshaped_test_data)
    print("reconstructing data")
    encoded_data = autoencoder.encoder(reshaped_test_data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()

    plottingManager = PlottingManager(
        draw_plots=draw_plots,
        draw_reconstructions=set_draw_reconstructions(draw_reconstructions, num_columns),
        num_to_show=num_to_show
    )

    plottingManager.plot_reconstructions(original_test_data, decoded_data.reshape(1, -1, num_columns)[0], column_names)

    print("calculating stats of all datapoints")

    plottingManager.plot_model_loss_val_loss(history)

    train_loss, test_loss, threshold = calculate_loss_and_threshold(autoencoder,
                                                                    reshaped_train_data, reshaped_test_data)

    plottingManager.plot_loss_histograms(train_loss, test_loss, threshold)
    plottingManager.plot_loss_bar_chart(test_loss, threshold)

    predict_anomalies(autoencoder, reshaped_test_data, threshold, date_time_series)


if __name__ == "__main__":
    conv_ae()
