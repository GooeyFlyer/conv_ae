import unittest

import tensorflow as tf
import numpy as np
import pandas as pd
import numpy.testing as npt

import src.get_data
import src.LossThresholdCalculator


class TestGetData(unittest.TestCase):
    data = pd.DataFrame({
        "Date_Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "param1": [2, 3, 4, 7, 6, 3, 2, 9, 1, 4],
        "param2": [3, 6, 8, 2, 1, 0, 5, 3, 2, 5],
        "param3": [7, 3, 2, 6, 5, 9, 8, 7, 1, 6]
    })

    def test_extend_data(self):
        """testing src.get_data.extend_data"""
        input_neurons = 4
        extended = src.get_data.extend_data(self.data, input_neurons)
        expected = pd.DataFrame({
            "Date_Time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10],
            "param1": [2, 3, 4, 7, 6, 3, 2, 9, 1, 4, 4, 4],
            "param2": [3, 6, 8, 2, 1, 0, 5, 3, 2, 5, 5, 5],
            "param3": [7, 3, 2, 6, 5, 9, 8, 7, 1, 6, 6, 6]
        })
        npt.assert_array_equal(expected, extended)

    def test_split_data_by_test_config(self):
        """testing src.get_data.split_data_by_test_config"""

        config_values = {
            "parameters": self.data.columns,
            "test_data_config": 0.6
        }
        train = pd.DataFrame({
            "Date_Time": [1, 2, 3, 4, 5, 6],
            "param1": [2, 3, 4, 7, 6, 3],
            "param2": [3, 6, 8, 2, 1, 0],
            "param3": [7, 3, 2, 6, 5, 9]
        })
        test = pd.DataFrame({
            "Date_Time": [7, 8, 9, 10],
            "param1": [2, 9, 1, 4],
            "param2": [5, 3, 2, 5],
            "param3": [8, 7, 1, 6]
        })

        print("test_data_config: 0.6")
        config_values["test_data_config"] = 0.6
        train_split, test_split = src.get_data.split_by_test_data_config(self.data, config_values)
        npt.assert_array_equal(train.values, train_split.values)
        npt.assert_array_equal(test.values, test_split.values)

        print("test_data_config: 8 - remember it's file index")
        config_values["test_data_config"] = 8
        train_split, test_split = src.get_data.split_by_test_data_config(self.data, config_values)
        npt.assert_array_equal(train.values, train_split.values)
        npt.assert_array_equal(test.values, test_split.values)

        print("test_data_config: None")
        config_values["test_data_config"] = None
        train_split, test_split = src.get_data.split_by_test_data_config(self.data, config_values)
        npt.assert_array_equal(self.data.values, train_split.values)
        npt.assert_array_equal(self.data.values, test_split.values)
        npt.assert_array_equal(train_split.values, test_split.values)

    def test_process_data_scaling(self):
        # testing src.get_data.process_data_scaling

        # need strings for date
        test_data = pd.DataFrame({
            "Date_Time": ["01-01-26", "02-01-26", "03-01-26", "04-01-26"],
            "param1": [10, 4, 0, 10],
            "param2": [2, 7, 10, 8]
        })

        expected = test_data.copy()
        expected["param1"] = [1, 0.4, 0, 1]
        expected["param2"] = [0, 0.625, 1, 0.75]

        df_scaled = src.get_data.process_data_scaling(test_data)

        npt.assert_array_equal(expected.values, df_scaled.values)


class TestLossThresholdCalculator(unittest.TestCase):

    calc = src.LossThresholdCalculator.LossThresholdCalculator("mean_absolute_error", (0.95, 0.99))

    def test_calculate_abs_error_per_channel(self):
        """testing src.LossThresholdCalculator.calculate_abs_error_per_channel"""

        # shape (data_points -> rows, channels -> cols)
        original = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])
        recon = np.array([[1, 3, 3],
                         [4, 5, 6],
                         [6, 8, 10]])

        expected = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 1]])
        result = self.calc.calculate_abs_error_per_channel(original, recon)

        npt.assert_array_equal(expected, result)

    def test_calculate_contribution_error(self):
        """testing src.LossThresholdCalculator.calculate_contribution_errors"""

        # shape (data_points -> rows, channels -> cols)
        abs_errors = np.array([[2., 6., 2.],
                               [5., 5., 10.],
                               [3., 4., 1.]])

        expected = np.array([[2/10, 6/10, 2/10],
                            [5/20, 5/20, 10/20],
                            [3/8, 4/8, 1/8]])

        result = self.calc.calculate_contribution_errors(abs_errors)

        npt.assert_array_equal(expected, result)

    def test_calculate_status_from_loss(self):
        """testing src.LossThresholdCalculator.calculate_status_from_loss"""

        expected = [1, 2, 2, 3, 3]

        loss = tf.convert_to_tensor([1, 3, 4, 5, 5.1])
        result = self.calc.calculate_status_from_loss(loss, (3, 5))

        npt.assert_array_equal(expected, result)


def load_tests(loader: unittest.TestLoader, standard_tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestGetData))
    suite.addTests(loader.loadTestsFromTestCase(TestLossThresholdCalculator))
    return suite


if __name__ == "__main__":
    unittest.main()
