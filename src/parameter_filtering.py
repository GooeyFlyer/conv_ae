import pandas as pd
import os

from src.load_options import load_yaml


def row_boolean(parameter_column: pd.Series, column_choice: str):
    """
    returns True for all if column_choice is None,
    or returns boolean for each item in parameter_lookup column that matches column_choice
    """
    if column_choice is None:
        return [True] * len(parameter_column)

    return parameter_column == column_choice


def boolean_all_columns(column_names_array: list[str], parameter_lookup: pd.DataFrame, config_values: dict
                        ) -> list[bool]:
    """returns boolean array for each item in parameter_lookup, if it matches settings in config_values"""
    total = [True] * len(parameter_lookup)

    for column_name in column_names_array:
        # pair up total and row_boolean result, then apply a and b for each pair
        total = [a and b for a, b in zip(
            total, row_boolean(parameter_lookup[column_name], config_values[column_name])
        )]

    return total


def pretty_column(setting: str | None) -> str:
    """"All" if setting is None, else setting"""
    if setting is None:
        return "All"
    return setting


def parameter_filtering(data: pd.DataFrame, config_values: dict) -> tuple[pd.DataFrame, str]:
    """find the parameters the user wants
    Returns:
        filtered dataframe
        pretty message for what settings the user chose, and what parameters that gives"""

    lookup_file_name = "parameter_lookup.csv"

    if not(os.path.exists(lookup_file_name)):
        print(f"WARNING: {lookup_file_name} not found, modelling all parameters")
        return data, ""

    parameter_lookup = pd.read_csv(lookup_file_name)

    # set(parameter_lookup[column_name].values)

    columns = ["GroupSystem", "System", "Subsystem", "Component", "Sensor"]

    filtered_lookup = parameter_lookup.loc[boolean_all_columns(
        columns,
        parameter_lookup, config_values
    )]

    filtered_parameters = filtered_lookup["ParName"]  # get just ParName
    filtered_parameters = list(set(filtered_parameters.values))  # remove duplicate parameters

    # creates pretty string of value of each column
    pretty_columns = ", ".join([f"{column_choice}={pretty_column(config_values[column_choice])}"
                                for column_choice in columns])

    if len(filtered_parameters) == 0:
        raise Exception(f"No parameters found for {pretty_columns}")

    message = f"""
filtered to: {pretty_columns}
included parameters: {" ".join(filtered_parameters)}
"""

    filtered_parameters.insert(0, "Date_Time")

    return data[filtered_parameters], message  # return only columns in filtered_parameters


if __name__ == "__main__":
    os.chdir("../")
    c = load_yaml("configuration.yml")  # ENV
    d = pd.read_csv(c["train_file_path"], sep=";")
    d, m = parameter_filtering(d, c)
    print(m)
    print(d.head())
