import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import layers, losses
from keras.models import Model

from data.get_data import *


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


def conv_ae(file_path: str):
    """
    split data, normalise data, build model, train model, reconstruct test_data, plot test_data against reconstruct
    """

    # split & normalise data
    train_data, test_data = process_data_scaling(file_path)

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
        plt.plot((test_data[:, x])[:num_to_show], label=f"test {x}", color="b")
        plt.plot((decoded_data[:, x])[:num_to_show], label=f"reconstructed {x}", color="r")
        plt.fill_between(np.arange(len((test_data[:, x])[:num_to_show])), (decoded_data[:, x])[:num_to_show],
                         (test_data[:, x])[:num_to_show], label="error", color="lightcoral")

        ax.set_yticks(np.linspace(0, 1, 11))
        plt.xlabel("Timestamps")
        plt.ylabel("Normalised values")
        ax.legend()
        plt.savefig(f"images/plot_{x}.png")
        plt.cla()

    print("plots saved to images/")


if __name__ == "__main__":
    clear_images_folder()
    conv_ae("data/FeatureDataSel.csv")
