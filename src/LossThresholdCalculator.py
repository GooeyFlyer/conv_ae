import tensorflow as tf
import keras
import numpy as np
import logging
logger = logging.getLogger("simple_logger")


class LossThresholdCalculator:
    def __init__(self, loss: str, threshold_quantiles: tuple[float, float]):
        """
        Parameters:
            loss (str): choice of loss function. See README for details
            threshold_quantiles (tuple[float, float]): tuple of quantile decimals between 0 and 1,
                that thresholds will be set at
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

        for x in threshold_quantiles:
            if not(0 <= x <= 1):
                raise ValueError(f"threshold_quantiles ({threshold_quantiles}) must be in the range [0, 1]")

        self.threshold_quantiles = threshold_quantiles

    def __call__(self, train_reconstructions: tf.Tensor, test_reconstructions: tf.Tensor,
                 reshaped_train_data: tf.Tensor, reshaped_test_data: tf.Tensor,
                 *args, **kwargs) -> tuple[tf.Tensor, tf.Tensor, tuple[float, float]]:
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
            thresholds (tuple[float]): tuple of thresholds calculated from train_loss, used for anomaly labelling
        """

        train_loss = tf.reshape(
            self.calculate_loss(y_pred=train_reconstructions, y_true=reshaped_train_data),
            shape=[-1]
        )

        threshold_s2 = np.quantile(train_loss, self.threshold_quantiles[0])

        threshold_s3 = np.quantile(train_loss, self.threshold_quantiles[1])
        logger.info(f"threshold status 2: {threshold_s2:.4f}, status 3: {threshold_s3:.4f}")

        test_loss = tf.reshape(
            self.calculate_loss(y_pred=test_reconstructions, y_true=reshaped_test_data),
            shape=[-1]
        )

        return train_loss, test_loss, (threshold_s2, threshold_s3)

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

    def calculate_status_from_loss(self, loss: tf.Tensor, threshold_values: tuple[float, float]) -> list[int]:
        """
        returns an array of size loss, with values 1-3, indicating status
        Parameters:
            loss (tf.Tensor): loss tensor
            threshold_values (tuple[float, float]): tuple of actual (not percentage) thresholds for status 2 and 3
        """

        status_array = []

        for x in loss:
            if x < threshold_values[0]:
                status_array.append(1)
            elif threshold_values[0] <= x < threshold_values[1]:
                status_array.append(2)
            elif threshold_values[1] <= x:
                status_array.append(3)
            else:
                raise ValueError(f"{x} in loss array broken")

        return status_array

    def contribution_axis(self, axis_values: np.ndarray) -> np.ndarray:
        """contribution error of single datapoint"""

        total = np.sum(axis_values)
        return axis_values / total

    def calculate_contribution_errors(self, errors: np.ndarray[np.ndarray]) -> np.ndarray[np.ndarray]:
        """takes a list of multiple error numpy arrays, returns 2d numpy, normalised to contribution errors
        data shape (data_points, channels)"""

        for index in range(errors.shape[0]):  # for every datapoint
            errors[index, :] = self.contribution_axis(errors[index, :])  # datapoint

        return errors

    def calculate_abs_error_per_channel(self, original_data: np.ndarray[np.ndarray], flat_recons: np.ndarray[np.ndarray]
                                        ) -> np.ndarray[np.ndarray]:
        """returns 2D array of abs error between 2D arrays original and recon, for each parameter
        data shape (data_points, channels)"""

        errors = np.zeros(original_data.shape)
        for index in range(original_data.shape[1]):  # for every channel
            org_column = original_data[:, index]
            recon_column = flat_recons[:, index]

            errors[:, index] = np.abs(org_column - recon_column)

        return errors
