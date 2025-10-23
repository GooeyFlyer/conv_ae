import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from keras import layers
from keras.models import Model

from MVD_reconstrucion.get_data import *


num_to_show = 50


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


def clear_images_folder():
    import os
    folder = "images"
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def draw_loss_histogram(ax: plt.Axes, loss, threshold: np.ndarray, title: str, max_loss: float):
    """Draws histogram with set styling for test and train loss"""

    ax.hist(loss, bins=50, label=f"{title} loss")
    ax.axvline(x=threshold, color="r", label="anomaly threshold")
    ax.set_xlim(0, max_loss)
    ax.set_ylabel("No of examples")
    ax.legend()


def predict(model, data, threshold):
    reconstructions = model(data)
    loss = keras.losses.mean_absolute_error(y_true=data, y_pred=reconstructions)
    return tf.math.less(loss, threshold)


def conv_ae(file_path: str):
    """
    split data, normalise data, build model, train model, reconstruct test_data, plot test_data against reconstruct
    """

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

    # plot test data against reconstructed data
    print("plotting test data against reconstructed data")
    for x in range(0, len(test_data[0])):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot((test_data[:, x])[:num_to_show], label=f"test {column_names[x]}", color="b")
        ax.plot((decoded_data[:, x])[:num_to_show], label=f"reconstructed {column_names[x]}", color="r")
        ax.fill_between(np.arange(len((test_data[:, x])[:num_to_show])), (decoded_data[:, x])[:num_to_show],
                         (test_data[:, x])[:num_to_show], label="error", color="lightcoral")

        # ax.set_xticklabels(date_time_series)
        ax.set_ylim(0, 1)
        ax.set_title(f"Plots of {column_names[x]}")
        ax.set_xlabel("Timestamps")
        ax.set_ylabel("Normalised values")
        ax.legend()
        fig.savefig(f"images/plot_{column_names[x]}.png")
        plt.close()

    print("plots saved to images/")

    # plot loss and val_loss
    print("plotting loss and val_loss")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation Loss")
    ax.legend()
    fig.savefig("images/stats/loss-val_loss.png")
    plt.close()

    # detect anomalies
    # reconstruction error for training data
    print("calculating test loss, threshold, & train loss")
    reconstructions = autoencoder.predict(train_data)
    train_loss = keras.losses.mean_absolute_error(y_pred=reconstructions, y_true=train_data)

    # choose threshold that is one standard deviation above the mean
    threshold = np.mean(train_loss) + np.std(train_loss)
    print("calculated anomaly Threshold: ", threshold)

    # reconstruction error for test data
    reconstructions = autoencoder.predict(test_data)
    test_loss = keras.losses.mean_absolute_error(y_pred=reconstructions, y_true=test_data)

    max_loss = round(np.max(tf.concat([train_loss, test_loss], axis=0)), 2) + 0.01
    print(max_loss)

    print("plotting loss histograms")
    # draw_histogram(train_loss, threshold, "Train", max_loss)
    # draw_histogram(test_loss, threshold, "Test", max_loss)

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(2, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("loss frequency")
    draw_loss_histogram(ax1, train_loss, threshold, "Train", max_loss)

    ax2 = fig.add_subplot(gs[1, 0])
    draw_loss_histogram(ax2, test_loss, threshold, "Test", max_loss)

    fig.savefig(f"images/stats/Test_Train_Loss.png")
    plt.close()

    print("stats plots saved to images/stats/")

    # print("\nstats:")
    # preds = predict(autoencoder, test_data, threshold)
    # print(preds)


if __name__ == "__main__":
    clear_images_folder()
    conv_ae("data/FeatureDataSel.csv")
