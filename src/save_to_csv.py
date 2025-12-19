import os
import pandas as pd
import logging
logger = logging.getLogger("simple_logger")


def save_to_csv(file_path: str, org_test, flat_test_recons, test_cont_errors,
                test_abs_errors, test_loss, test_status, test_Date_Time, channel_names):

    df = pd.DataFrame()

    df["Date_Time"] = test_Date_Time
    df["loss"] = test_loss
    df["status"] = test_status

    for index in range(org_test.shape[1]):  # for every channel
        channel_name = channel_names[index]

        df[channel_name + "_original"] = org_test[:, index]
        df[channel_name + "_reconstructed"] = flat_test_recons[:, index]
        df[channel_name + "_absolute_error"] = test_abs_errors[:, index]
        df[channel_name + "_contribution_error"] = test_cont_errors[:, index]

    os.makedirs(os.path.split(file_path)[0], exist_ok=True)

    df.to_csv(file_path, index_label="Index")
    logger.info(f"data saved to {file_path}")
