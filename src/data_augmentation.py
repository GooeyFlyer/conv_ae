import numpy as np
import scipy.interpolate
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from src.get_data import process_data_scaling


# def augment_gaussian(x: np.ndarray) -> np.ndarray:
#     noisy = np.random.normal(x, 0.05)
#     scalar = MinMaxScaler(feature_range=(0, 1))
#     scaled = scalar.fit_transform(noisy)
#     return scaled


def augment_magnitude_warping(x: np.ndarray):
    """code from tsgm https://gitub.com/AlexanderVNikitin/tsgm"""

    n_timesteps = x.shape[0]
    n_features = x.shape[1]

    n_knots = 4

    orig_steps = np.arange(n_timesteps)
    random_warps = np.random.normal(loc=1, scale=0.20, size=(n_knots + 2, n_features))
    warp_steps = (
            np.ones((n_features, 1)) * (np.linspace(0, n_timesteps-1, num=n_knots + 2))
    ).T

    warper = np.array([
        # creates CubicSpline warper line function, then calls orig_steps to get values, for each feature (dim)
        scipy.interpolate.CubicSpline(
            warp_steps[:, dim], random_warps[:, dim]
        )(orig_steps)
        for dim in range(n_features)
    ]).T
    print(warper.shape)
    result = x * warper

    scalar = MinMaxScaler(feature_range=(0, 1))
    scaled = scalar.fit_transform(result)
    return scaled, warper


if __name__ == "__main__":
    array, _, channel_names = process_data_scaling("../data/FeatureDataSel.csv")
    array = np.delete(array, [17,16,15,14,13,12,11,10,9,8,7,6,5,4,3], axis=1)

    # wind = augment_window_warping(array)
    mag, warper = augment_magnitude_warping(array)

    print(f"\nmeans: {np.mean(array)} {np.mean(mag)}")

    total = array

    for x in range(0, total.shape[1]):  # for every channel
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot((total[:, x]), label=f"Array", color="b")
        ax.plot((mag[:, x]), label=f"Warped array", color="r")
        ax.plot((warper[:, x]), label=f"Warper", color="g")
        ax.fill_between(
            np.arange(len((array[:, x]))),
            (mag[:, x]),
            (array[:, x]),
            label="error", color="lightcoral"
        )

        # ax.set_xticklabels(date_time_series)
        ax.set_title(f"Plots of {channel_names[x]}")
        ax.set_ylim(0)
        ax.set_xlabel("Timestamps")
        ax.set_ylabel("Normalised values")
        ax.legend()
        ax.grid(True)

    plt.show()
