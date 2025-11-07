from src.conv_ae import anomaly_detection
from src.load_options import load_yaml
import pandas as pd


def main():
    config_values = load_yaml("configuration.yml")  # ENV values
    data = pd.read_csv(config_values["train_file_path"], sep=";")

    # data, message = parameter_filtering(data, config_values)

    anomaly_detection(data, config_values, "no parameter filtering")


if __name__ == "__main__":
    main()
