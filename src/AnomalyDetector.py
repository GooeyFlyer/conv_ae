import tensorflow as tf
from keras import Sequential, layers
from keras.models import Model
from keras.utils import plot_model
import numpy as np


class AnomalyDetector(Model):
    def __init__(self, steps_in_batch: int, num_columns: int):
        super(AnomalyDetector, self).__init__()

        strides = 1  # by how many datapoints the filter will move as it slides over input.
        pool_size = 4
        kernel_size = 7
        activation = "relu"

        # down-samples and learns spatial features
        self.encoder = Sequential([
            layers.Input((steps_in_batch, num_columns)),

            # convolve each window in input with each filter
            layers.Conv1D(
                filters=num_columns, kernel_size=kernel_size, strides=strides, activation=activation, padding="same"
            ),

            # reduces dimensionality of input by factor pool_size
            layers.MaxPool1D(pool_size=pool_size, padding="same"),

            layers.Conv1D(
                filters=num_columns*2, kernel_size=kernel_size, strides=strides, activation=activation, padding="same"
            ),
            layers.MaxPool1D(pool_size=pool_size, padding="same")
        ], name="encoder")

        # down-samples and learns spatial features
        self.decoder = Sequential([
            layers.Input((steps_in_batch//(pool_size*pool_size) + steps_in_batch % (pool_size*pool_size), num_columns*2)),

            # expands dimensionality of input by factor pool_size
            layers.UpSampling1D(size=pool_size),

            # reverse of convolution layer
            layers.Conv1DTranspose(
                filters=num_columns*2, kernel_size=kernel_size, strides=strides, activation=activation, padding="same"
            ),

            layers.UpSampling1D(size=pool_size),
            layers.Conv1DTranspose(
                filters=num_columns, kernel_size=kernel_size, strides=strides, activation=activation, padding="same"
            ),

            # squeezes values between 0 and 1
            layers.Dense(units=num_columns, activation="sigmoid")
        ], name="decoder")

    def call(self, x: np.ndarray) -> tf.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    # simple usage

    # batche size, datapoints in batch, features
    data = np.random.rand(100, 16, 18)

    # build model
    autoencoder = AnomalyDetector(data.shape[1], data.shape[2])
    autoencoder.compile(optimizer="adam", loss="mae")
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    plot_model(autoencoder.encoder, to_file="../models/encoder.png", show_shapes=True, show_layer_names=True)
    plot_model(autoencoder.decoder, to_file="../models/decoder.png", show_shapes=True, show_layer_names=True)

    history = autoencoder.fit(
        data, data,
        epochs=20,
        shuffle=False,
        verbose=2
    )

    encoded_data = autoencoder.encoder(data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()
