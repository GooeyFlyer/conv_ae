from src.conv_ae import anomaly_detection
from src.load_options import load_yaml
from src.parameter_filtering import parameter_filtering
import pandas as pd


def main():
    config_values = load_yaml("configuration.yml")  # ENV values
    data = pd.read_csv(config_values["train_file_path"], sep=";")

    if "GroupSystem" in config_values:
        data, message = parameter_filtering(data, config_values)
    else:
        message = "no parameter filtering"

    print(message)

    anomaly_detection(data, config_values, message)


if __name__ == "__main__":
    main()
