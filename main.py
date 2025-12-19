import sys

from src.conv_ae import anomaly_detection
from src.load_options import load_yaml
from src.get_data import process_data_scaling, prepare_DataFrames
import pandas as pd
import logging


def setup_logging():
    logger = logging.getLogger("simple_logger")
    logger.setLevel(logging.DEBUG)  # determines level of log to pass to Handlers

    formatter = logging.Formatter("%(levelname)s %(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)  # determines level of log the handler will send on
    ch.setFormatter(formatter)

    file_logger = logging.FileHandler("logs.log", mode="w")
    file_logger.setLevel(logging.INFO)
    file_logger.setFormatter(formatter)

    logger.addHandler(file_logger)
    logger.addHandler(ch)


def main():
    logger.info("starting")
    config_values = load_yaml("configuration.yml")  # ENV values
    data = pd.read_csv(config_values["train_file_path"], sep=";")

    # scales data
    data = process_data_scaling(data)

    # split and extend data
    train_df, test_df = prepare_DataFrames(data, config_values)

    del data

    anomaly_detection(train_df, test_df, config_values)

    logger.info("finished")


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("simple_logger")
    main()
