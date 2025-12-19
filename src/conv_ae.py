import numpy as np
import tensorflow as tf
import pandas as pd

from src.PlottingManager import PlottingManager
from src.AnomalyDetector import AnomalyDetector
from src.LossThresholdCalculator import LossThresholdCalculator
from src.save_to_csv import save_to_csv

import logging
logger = logging.getLogger("simple_logger")


def set_draw_reconstructions(draw_reconstructions: str, num_columns: int) -> bool:
    """
    returns boolean for if reconstructions should be drawn, depending on draw_reconstructions:
    'yes', 'no', or 'auto'
    'auto' draws if num_columns <= 20
    """
    try:
        return {"yes": True,
                "no": False,
                "auto": num_columns <= 20,
                True: True,
                False: False}[draw_reconstructions]
    except KeyError:
        logger.error(f"draw_reconstructions is invalid value {draw_reconstructions}.\nDefaulting to False")
        return False


def anomaly_detection(train_df: pd.DataFrame, test_df: pd.DataFrame, config_values: dict):
    """
    normalise data, split data, build model, train model, reconstruct test_data, plot graphs, find anomalies
    """

    train_Date_Time = train_df.pop("Date_Time")
    test_Date_Time = test_df.pop("Date_Time")

    logger.info(f"train shape: {train_df.shape[0]} datapoints {train_df.shape[1]} parameters")
    logger.info(f"test shape: {test_df.shape[0]} datapoints {test_df.shape[1]} parameters")

    channel_names = train_df.columns.tolist()
    num_channels = len(channel_names)

    original_train_data = train_df.to_numpy()
    original_test_data = test_df.to_numpy()

    # batch shape, steps_in_batch, num features
    reshaped_train_data = original_train_data.reshape((-1, config_values["input_neurons"], num_channels))
    reshaped_test_data = original_test_data.reshape((-1, config_values["input_neurons"], num_channels))

    # raw_scaled_data, date_time_series, channel_names = process_data_scaling(data)

    logger.info(f"modelling on {num_channels} parameters:")

    verbose = {True: "auto", False: 0}[config_values["verbose_model"]]

    # LossThresholdCalculator initialised here, as it contains error checking for config_values["threshold_s2_quantile"]
    calc = LossThresholdCalculator(config_values["loss"],
                                   (config_values["threshold_s2_quantile"], config_values["threshold_s3_quantile"]))

    # build model
    logger.info("building model")
    autoencoder = AnomalyDetector(
        num_input_neurons=config_values["input_neurons"],
        num_features=num_channels,
        strides=config_values["strides"],
        pool_size=config_values["pool_size"],
        kernel_size=config_values["kernel_size"],
        activation=config_values["activation"]
    )
    autoencoder.compile(optimizer=config_values["optimizer"], loss=config_values["loss"], metrics=["accuracy"])

    if config_values["verbose_model"]:
        autoencoder.encoder.summary()
        autoencoder.decoder.summary()

    # train model
    logger.info("training model")
    history = autoencoder.fit(
        reshaped_train_data, reshaped_train_data,
        epochs=config_values["epochs"],
        validation_data=(reshaped_test_data, reshaped_test_data),
        shuffle=False,
        verbose=verbose
    )

    logger.info("reconstructing data")
    train_reconstructions = autoencoder.predict(
        reshaped_train_data,
        batch_size=reshaped_train_data.shape[0],
        verbose=verbose
    )  # tf.Tensor
    test_reconstructions = autoencoder.predict(
        reshaped_test_data,
        batch_size=reshaped_test_data.shape[0],
        verbose=verbose
    )  # tf.Tensor

    logger.info("calculating stats of all datapoints")

    flat_train_recons = train_reconstructions.reshape(1, -1, num_channels)[0]  # flatten array
    flat_test_recons = test_reconstructions.reshape(1, -1, num_channels)[0]  # flatten array

    test_abs_errors = calc.calculate_abs_error_per_channel(original_test_data, flat_test_recons)

    test_cont_errors = calc.calculate_contribution_errors(test_abs_errors)

    train_loss, test_loss, thresholds = calc(train_reconstructions, test_reconstructions,
                                             reshaped_train_data, reshaped_test_data)

    train_status = calc.calculate_status_from_loss(train_loss, thresholds)
    test_status = calc.calculate_status_from_loss(test_loss, thresholds)

    logger.info("saving data")
    save_to_csv(f"results/calculated_data.csv", original_test_data, flat_test_recons, test_cont_errors,
                test_abs_errors, test_loss, test_status, test_Date_Time, channel_names)

    del reshaped_train_data, test_abs_errors, train_reconstructions, test_reconstructions

    # write_anomalies(calc, test_reconstructions, reshaped_test_data, threshold, date_time_series, filter_message)

    logger.info("plotting")

    num_channels = original_train_data.shape[1]

    plottingManager = PlottingManager(
        draw_plots=config_values["draw_plots"],  # decides if images are drawn
        draw_reconstructions=set_draw_reconstructions(config_values["draw_reconstructions"], num_channels),
        error_plot=config_values["error_plot"],
        verbose=False
    )

    plottingManager.plot_reconstructions(
        "train",
        original_train_data,
        flat_train_recons,  # flatten array
        loss=train_loss, column_names=channel_names
    )
    plottingManager.plot_reconstructions(
        "test",
        original_test_data,
        flat_test_recons,  # flatten array
        loss=test_loss, column_names=channel_names
    )
    plottingManager.plot_reconstructions(
        "combined",
        np.concatenate((original_train_data, original_test_data)),
        np.concatenate((flat_train_recons, flat_test_recons)),
        loss=tf.concat([train_loss, test_loss], axis=0), column_names=channel_names
    )
    del flat_train_recons, flat_test_recons, original_train_data, original_test_data

    plottingManager.plot_contribution_errors(test_cont_errors, channel_names)

    del test_cont_errors

    plottingManager.plot_model_loss_val_loss(history)

    del history

    plottingManager.plot_loss_histograms(train_loss, test_loss, thresholds)

    plottingManager.plot_loss_line_chart("train", train_loss, train_status, thresholds)
    plottingManager.plot_loss_line_chart("test", test_loss, test_status, thresholds)

    # plottingManager.plot_zoomed_loss_line_chart("train", train_loss, threshold)
    # plottingManager.plot_zoomed_loss_line_chart("test", test_loss, threshold)
