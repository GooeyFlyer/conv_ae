import yaml


def load_yaml(file_name: str) -> dict:
    try:
        with open(file_name, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find {file_name}")

    return verify_yaml_values(data)


def verify_yaml_values(data: dict) -> dict:
    """Makes sure all values in pdf_configuration.yml are valid
    Raises error if any values are invalid.
    Parameters:
        data (dict): dictionary from load_yaml()
    Returns:
        data (dict): verified dictionary
    """

    # add any new settings to this dictionary (and their expected type)
    # a setting here does not have to be in the data dictionary
    setting_types = {
        "file_path": str,
        "draw_plots": bool,
        "draw_reconstructions": str,
        "num_to_show": int,
    }

    for key, value in data.items():
        value_type = setting_types[key]

        # noinspection PyTypeHints
        if not isinstance(value, value_type):  # Not recognised as a valid argument by PyCharm. Ignore the error.
            raise ValueError(key + " in pdf_configuration.yml must be a " + value_type.__name__)

        # special cases handled below

        elif key in ["num_to_show"]:
            if value <= 0:
                raise ValueError(key + " in configuration.yml must be more that 1")

        elif key == "draw_reconstructions":
            if value not in ["yes", "no", "auto"]:
                raise ValueError(key + " in configuration.yml must be 'yes', 'no', or 'auto'")

    return data


if __name__ == "__main__":
    data_list = load_yaml("../configuration.yml")
    print(data_list)
