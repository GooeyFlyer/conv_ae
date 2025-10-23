import keras

from keras import layers
from keras.models import Model

from MVD_reconstrucion.get_data import *
from MVD_reconstrucion.PlottingManager import PlottingManager


class AnomalyDetector(Model):
    def __init__(self, columns: int):
        super(AnomalyDetector, self).__init__()
        self.encoder = keras.Sequential([
            layers.Dense(columns, activation="relu"),
            layers.Dense(8, activation="relu"),
            layers.Dense(4, activation="relu"),
        ])
        self.decoder = keras.Sequential([
            layers.Dense(8, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(columns, activation="sigmoid"),
        ])

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

    print(f"{len(test_data[0])} columns in data")

    # build model
    print("building model")
    autoencoder = AnomalyDetector(len(test_data[0]))
    autoencoder.compile(optimizer="adam", loss="mae")
    autoencoder.summary()

    # train model
    print("training model")
    history = autoencoder.fit(train_data, train_data, epochs=20, validation_data=(test_data, test_data), shuffle=True)

    # reconstructing test_data
    print("reconstructing data")
    encoded_data = autoencoder.encoder(test_data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()

    plottingManager.plot_reconstructions(test_data, decoded_data, column_names)

    print("calculating stats of all datapoints")

    plottingManager.plot_model_loss_val_loss(history)

    # anomaly detection
    # reconstruction error for training data
    print("calculating test loss, threshold, & train loss")
    reconstructions = autoencoder.predict(train_data)  # reconstructs training data (remember contains anomalies)
    train_loss = keras.losses.mean_absolute_error(y_true=reconstructions, y_pred=train_data)

    # choose threshold that is one standard deviation above the mean
    threshold = float(np.mean(train_loss) + np.std(train_loss))
    print("calculated anomaly Threshold: ", threshold)

    # reconstruction error for test data
    reconstructions = autoencoder.predict(test_data)  # reconstructs testing data (remember contains anomalies)
    test_loss = keras.losses.mean_absolute_error(y_true=reconstructions, y_pred=test_data)

    plottingManager.plot_loss_histograms(train_loss, test_loss, threshold)
    plottingManager.plot_loss_bar_chart(test_loss, threshold)

    predictions = predict(autoencoder, test_data, threshold)  # 1 prediction per test_data datapoint (a.k.a. timestamp)
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
