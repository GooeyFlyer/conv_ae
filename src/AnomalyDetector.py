import keras
from keras import layers
from keras.models import Model
import tensorflow as tf
import numpy as np


class AnomalyDetector(Model):
    def __init__(self, num_in_batch: int, num_columns: int):
        super(AnomalyDetector, self).__init__()

        strides = 1  # by how many datapoints the filter will move as it slides over input.
        pool_size = 2

        # down-samples and learns spatial features
        self.encoder = keras.Sequential([
            layers.Input((num_in_batch, num_columns)),

            # convolve each window in input with each filter
            layers.Conv1D(filters=num_columns, kernel_size=3, strides=strides, activation="relu", padding="same"),

            # reduces dimensionality of input by factor pool_size
            layers.MaxPool1D(pool_size=pool_size, padding="same"),

            layers.Conv1D(filters=num_columns*2, kernel_size=3, strides=strides, activation="relu", padding="same"),
            layers.MaxPool1D(pool_size=pool_size, padding="same")
        ], name="encoder")

        # down-samples and learns spatial features
        self.decoder = keras.Sequential([
            layers.Input((num_in_batch//pool_size//pool_size, 36)),

            # expands dimensionality of input by factor pool_size
            layers.UpSampling1D(size=pool_size),

            # reverse of convolution layer
            layers.Conv1DTranspose(filters=num_columns*2, kernel_size=3, strides=strides, activation="relu", padding="same"),

            layers.UpSampling1D(size=pool_size),
            layers.Conv1DTranspose(filters=num_columns, kernel_size=3, strides=strides, activation="relu", padding="same"),

            # squeezes values between 0 and 1
            layers.Dense(units=num_columns, activation="sigmoid")
        ], name="decoder")

    def call(self, x: np.ndarray) -> tf.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # simple usage

    # 50 batches, 8 datapoints in batch, 18 features
    data = np.random.rand(50, 8, 18)

    # build model
    print("only building model")
    autoencoder = AnomalyDetector(data.shape[1], data.shape[2])
    autoencoder.compile(optimizer="adam", loss="mae")
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    history = autoencoder.fit(
        data, data,
        epochs=5,
        batch_size=None,
        shuffle=True
    )

    encoded_data = autoencoder.encoder(data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()
