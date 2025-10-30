import yaml


def load_yaml(file_name: str) -> dict:
    try:
        with open(file_name, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)

    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot find {file_name}")

    return verify_yaml_values(data)


def expected_type_array_as_string(expected_type_array: list) -> str:
    """returns string of type names"""
    a = [expected_type.__name__ for expected_type in expected_type_array]
    return ", ".join(a)


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
        "train_file_path": [str],
        "test_data_config": [str, type(None), int],
        "draw_plots": [bool],
        "draw_reconstructions": [str],
        "num_to_show": [int],
        "verbose_model": [bool],
    }

    for key, value in data.items():
        expected_type_array = setting_types[key]

        valid = False
        for expected_type in expected_type_array:
            if isinstance(value, expected_type):
                valid = True

        if not valid:
            raise ValueError(
                key + " in configuration.yml must be a " + expected_type_array_as_string(expected_type_array)
            )

        # special cases handled below

        elif key in ["num_to_show"]:
            if value <= 0:
                raise ValueError(key + " in configuration.yml must at least 1")

        elif key == "draw_reconstructions":
            if value not in ["yes", "no", "auto"]:
                raise ValueError(key + " in configuration.yml must be 'yes', 'no', or 'auto'")

        elif key == "test_data_config":
            if isinstance(value, int):
                if value <= 2:
                    raise ValueError(key + " in configuration.yml must at least 3")

    return data


if __name__ == "__main__":
    data_list = load_yaml("../configuration.yml")
    print(data_list)
