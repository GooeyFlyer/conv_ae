import tensorflow as tf
import keras
import numpy as np


class LossThresholdCalculator:
    def __init__(self, loss: str, threshold_quantile: float):
        """
        Parameters:
            loss (str): choice of loss function. See README for details
            threshold_quantile (float): quantile decimal between 0 and 1, for threshold will be set at
        """

        # you can add your own functions here, ensure they conform to f(y_pred, y_true)
        loss_function_dict = {
            "mean_absolute_error": keras.losses.mean_absolute_error,
            "mean_squared_error": keras.losses.mean_squared_error,
            "mean_absolute_percentage_error": self.mean_absolute_percentage_error,
            "mean_squared_logarithmic_error": keras.losses.mean_squared_logarithmic_error,
            "cosine_similarity": self.cosine_similarity,
            "huber": keras.losses.huber,
            "log_cosh": keras.losses.log_cosh,
        }

        try:
            self.calculate_loss = loss_function_dict[loss]
            self.loss = loss

        except KeyError:
            raise KeyError(f"loss ({loss}) not found. Supported options are {loss_function_dict.keys()}")

        if 0 <= threshold_quantile <= 1:
            self.threshold_quantile = threshold_quantile

        else:
            raise ValueError(f"threshold_quantile ({threshold_quantile}) must be in the range [0, 1]")

    def __call__(self, train_reconstructions: tf.Tensor, test_reconstructions: tf.Tensor,
                 reshaped_train_data: tf.Tensor, reshaped_test_data: tf.Tensor,
                 *args, **kwargs) -> tuple[tf.Tensor, tf.Tensor, float]:
        """
        returns loss arrays with values between 0 and 1.
        Parameters:
            train_reconstructions (tf.Tensor): reconstructed data from autoencoder
            test_reconstructions (tf.Tensor):
            reshaped_train_data (tf.Tensor): original data, reshaped to batch_size, num_in_batch, channels
            reshaped_test_data (tf.Tensor):
        Returns:
            train_loss (tf.Tensor): loss for train data
            test_loss (tf.Tensor): loss for test data (anomaly detection split)
            threshold (float): the threshold calculated from train_loss, used for anomaly labelling
        """

        print("\ncalculating test loss, threshold, & train loss")
        print(f"using {self.loss} and threshold quantile {self.threshold_quantile}")

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

        # modify the result of keras loss functions to fit with PlottingManager

    def mean_absolute_percentage_error(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """scales result of keras mape to between 0 and 1"""
        loss = keras.losses.mean_absolute_percentage_error(y_true, y_pred)

        from sklearn.preprocessing import MinMaxScaler

        # feature scaling - ensure they all fall within 0 to 1
        scalar = MinMaxScaler(feature_range=(0, 1))
        return scalar.fit_transform(loss.numpy())

    def cosine_similarity(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """adds 1 to result of keras cosine_similarity"""
        loss = keras.losses.cosine_similarity(y_true, y_pred)
        return loss + 1
