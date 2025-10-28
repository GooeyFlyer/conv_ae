import keras
from keras import layers
from keras.models import Model


class AnomalyDetector(Model):
    def __init__(self, num_columns: int):
        super(AnomalyDetector, self).__init__()

        # down-samples and learns spatial features
        self.encoder = keras.Sequential([
            layers.Conv1D(8, kernel_size=3, activation="relu", name="conv1d_1", padding="same"),
            layers.ReLU(),  # returns input if input above 0
            layers.MaxPool1D(pool_size=1, padding="same"),
            layers.Conv1D(4, kernel_size=3, activation="relu", name="conv1d_2", padding="same"),
            layers.ReLU(),
            layers.MaxPool1D(pool_size=1, padding="same")
        ])

        # down-samples and learns spatial features
        self.decoder = keras.Sequential([
            layers.Conv1DTranspose(8, kernel_size=3, activation="relu", name="ctrans1d_1", padding="same"),
            layers.ReLU(),
            layers.Conv1DTranspose(16, kernel_size=3, activation="relu", name="ctrans1d_2", padding="same"),
            layers.ReLU(),
            layers.Dense(num_columns, activation="sigmoid")  # squeezes values between 0 and 1
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
