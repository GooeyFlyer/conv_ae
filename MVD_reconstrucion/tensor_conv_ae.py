import keras
import numpy as np

from keras import layers
from keras.models import Model

from MVD_reconstrucion.get_data import *
from MVD_reconstrucion.PlottingManager import PlottingManager

# TODO: Run trained model on combined train_data and test_data

# TODO: make sure implemented architecture conforms to this
# Autoencoder architecture: (class names from torch)
# encoder:
#     Conv2d(in=3, out=16, kernel=3),
#     ReLU(),
#     MaxPool2d(window_square_size = 2),
#     Conv2d(16, 8, 3, stride=1, padding=1),
#     ReLU(),
#     MaxPool2d(2, stride=2)
#
# decoder:
#     ConvTranspose2d(8, 16, 3),
#     ReLU(),
#     ConvTranspose2d(16, 3, 3),
#     Sigmoid()  # ensures output values are between 0 and 1


class AnomalyDetector(Model):
    def __init__(self, num_columns: int, architecture: str = "new"):
        super(AnomalyDetector, self).__init__()

        if architecture == "old":
            # down-samples and learns spatial features
            self.encoder = keras.Sequential([
                layers.Dense(num_columns, activation="relu"),
                layers.Dense(8, activation="relu"),
                layers.Dense(4, activation="relu"),
            ])

            # # down-samples and learns spatial features
            self.decoder = keras.Sequential([
                layers.Dense(8, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(num_columns, activation="sigmoid"),
            ])

        # down-samples and learns spatial features
        elif architecture == "new":
            self.encoder = keras.Sequential([
                layers.Conv1D(18, kernel_size=3, activation="relu", name="conv1d_1"),
                layers.ReLU(),
                layers.MaxPool1D(pool_size=1),
                layers.Conv1D(8, kernel_size=3, activation="relu", name="conv1d_2"),
                layers.ReLU(),
                layers.MaxPool1D(pool_size=1)
            ])

            # down-samples and learns spatial features
            self.decoder = keras.Sequential([
                layers.Conv1DTranspose(14, kernel_size=3, activation="relu", name="convt1d_1"),
                layers.ReLU(),
                layers.Conv1DTranspose(18, kernel_size=3, activation="relu", name="convt1d_2"),
                layers.ReLU(),
                layers.Dense(18, activation="sigmoid")
            ])

        else:
            print(f"""architecture parameter must be "new" or "old". Currently is {architecture}""")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def predict(model, test_data, threshold):
    """
    Returns:
        tensorflow array, of boolean if datapoint loss < threshold, for each datapoint (a.k.a timestamp) in test_data
    """
    reconstructions = model(test_data)  # test_data ran through trained model
    loss = keras.losses.mean_absolute_error(y_true=reconstructions, y_pred=test_data)
    return tf.math.less(loss, threshold)


def conv_ae(file_path: str, draw_plots: bool, num_to_show: int):
    """
    split data, normalise data, build model, train model, reconstruct test_data, plot graphs, find anomalies
    """

    plottingManager = PlottingManager(draw_plots=draw_plots, num_to_show=num_to_show)

    # split & normalise data
    train_data, test_data, date_time_series, column_names = process_data_scaling(file_path)

    # data sizes
    train_size = train_data.shape[0]
    test_size = test_data.shape[0]
    num_columns = train_data.shape[1]

    # batch_shape, steps, channels
    train_data = train_data.reshape(1, train_size, num_columns)
    test_data = test_data.reshape(1, test_size, num_columns)

    print(train_data.shape)  # (1, 1228, 18)

    # build model
    print("building model")
    autoencoder = AnomalyDetector(num_columns, "old")
    autoencoder.compile(optimizer="adam", loss="mae")
    autoencoder.summary()

    # train model
    print("training model")
    history = autoencoder.fit(train_data, train_data, epochs=50, validation_data=(test_data, test_data), shuffle=True)

    # reconstructing test_data
    print("reconstructing data")
    encoded_data = autoencoder.encoder(test_data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()

    print(decoded_data.shape)

    plottingManager.plot_reconstructions(test_data[0], decoded_data[0], column_names)

    print("calculating stats of all datapoints")

    plottingManager.plot_model_loss_val_loss(history)

    # anomaly detection
    # reconstruction error for training data
    print("calculating test loss, threshold, & train loss")
    reconstructions = autoencoder.predict(train_data)  # reconstructs training data (remember contains anomalies)
    train_loss = keras.losses.mean_absolute_error(y_true=reconstructions[0], y_pred=train_data[0])

    # choose threshold that is one standard deviation above the mean
    threshold = float(np.mean(train_loss) + np.std(train_loss))
    print("calculated anomaly Threshold: ", threshold)

    # reconstruction error for test data
    reconstructions = autoencoder.predict(test_data)  # reconstructs testing data (remember contains anomalies)
    test_loss = keras.losses.mean_absolute_error(y_true=reconstructions[0], y_pred=test_data[0])

    plottingManager.plot_loss_histograms(train_loss, test_loss, threshold)
    plottingManager.plot_loss_bar_chart(test_loss, threshold)

    predictions = predict(autoencoder, test_data, threshold)[0]  # 1 prediction per test_data datapoint (a.k.a. timestamp)
    print("predictions info: ", tf.size(predictions))

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
    with open("anomaly_stats.txt", "w") as file:
        file.write(output)

    print("\nstats saved to anomaly_stats.txt")


if __name__ == "__main__":
    conv_ae("data/FeatureDataSel.csv", draw_plots=True, num_to_show=50)
