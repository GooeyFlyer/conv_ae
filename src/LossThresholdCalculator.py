import tensorflow as tf
import keras
import numpy as np


class LossThresholdCalculator:
    def __init__(self, loss: str, threshold_quantile: float):

        # TODO: make these work with PlottingManager
        loss_function_dict = {
            "mean_absolute_error": keras.losses.mean_absolute_error,
            "mean_squared_error": keras.losses.mean_squared_error,
            # "mean_absolute_percentage_error": keras.losses.mean_absolute_percentage_error,
            "mean_squared_logarithmic_error": keras.losses.mean_squared_logarithmic_error,
            # "cosine_similarity": keras.losses.cosine_similarity,
            "huber": keras.losses.huber,
            "log_cosh": keras.losses.log_cosh,
            # "tversky": keras.losses.tversky,
            # "dice": keras.losses.dice
        }

        try:
            self.calculate_loss = loss_function_dict[loss]

        except KeyError as e:
            raise KeyError(f"loss ({loss}) not found. Supported options are {loss_function_dict.keys()}")

        if 0 <= threshold_quantile <= 1:
            self.threshold_quantile = threshold_quantile

        else:
            raise ValueError(f"threshold_quantile ({threshold_quantile}) must be in the range [0, 1]")

    def __call__(self, train_reconstructions, test_reconstructions, reshaped_train_data, reshaped_test_data,
                 *args, **kwargs):

        print("\ncalculating test loss, threshold, & train loss")

        train_loss = tf.reshape(
            self.calculate_loss(y_pred=train_reconstructions, y_true=reshaped_train_data),
            shape=[-1]
        )

        threshold = np.quantile(train_loss, self.threshold_quantile)
        print("calculated anomaly Threshold: ", threshold)

        test_loss = tf.reshape(
            self.calculate_loss(y_pred=test_reconstructions, y_true=reshaped_test_data),
            shape=[-1]
        )

        return train_loss, test_loss, threshold
