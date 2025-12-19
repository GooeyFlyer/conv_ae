import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
import os
import logging
logger = logging.getLogger("simple_logger")


class PlottingManager:
    def __init__(self, draw_plots: bool, draw_reconstructions: bool, error_plot: str,
                 verbose: bool = True):
        """
        Parameters:
            draw_plots (bool): decides if images are drawn
            draw_reconstructions (bool): decides if reconstruction plots are drawn
        """
        self.draw_plots = draw_plots
        self.verbose = verbose
        self.reconstructions_paths = {"train": f"results/plots/train_reconstructions",
                                      "test": f"results/plots/test_reconstructions",
                                      "combined": f"results/plots/combined_reconstructions"}
        self.stats_path = f"results/stats"
        self.draw_reconstructions = draw_reconstructions

        if error_plot in ["between", "floor"]:
            self.error_plot = error_plot
        else:
            if verbose:
                logger.warning("WARNING: error_plot not 'between' or 'floor', defaulting to 'between'")

        if self.draw_plots:
            for x in self.reconstructions_paths.values():
                os.makedirs(x, exist_ok=True)
                self.clear_images_folder(x)
                os.makedirs(self.stats_path, exist_ok=True)
            self.clear_images_folder(self.stats_path)

    def place_legend(self, ax: plt.Axes):
        box = ax.get_position()

        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # shrink width to 80%
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    def plot_reconstructions(self, split_name: str, original_data: np.ndarray, reconstructed_data: np.ndarray, loss,
                             column_names):
        """plot original data against reconstructed data"""

        if self.draw_plots and self.draw_reconstructions:
            logger.debug(f"plotting original {split_name} data against reconstructed data")
            folder_path = self.reconstructions_paths[split_name]

            for x in range(0, original_data.shape[1]):  # for every channel
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot((reconstructed_data[:, x]), label="reconstructed", color="r")
                ax.plot((original_data[:, x]), label=f"original", color="b")
                ax.plot(loss, label=f"{split_name} loss", color="g")

                if self.error_plot == "floor":
                    diff = abs(original_data[:, x] - reconstructed_data[:, x])
                    ax.plot(
                        diff, label="absolute error", color="lightcoral"
                    )

                elif self.error_plot == "between":
                    ax.fill_between(
                        np.arange(len((original_data[:, x]))),
                        (original_data[:, x]),
                        (reconstructed_data[:, x]),
                        label="absolute error", color="lightcoral"
                    )

                # ax.set_xticklabels(date_time_series)
                ax.set_ylim(0, 1)
                ax.set_title(f"Plots of {column_names[x]}")
                ax.set_xlabel("Timestamps")
                ax.set_ylabel("Normalised values")
                self.place_legend(ax)

                full_path = os.path.join(folder_path, f"plot_{column_names[x]}.png")
                self.save_fig(fig, full_path, private_verbose=False)

            logger.debug(f"plots saved to {folder_path}")

        elif self.draw_plots and not self.draw_reconstructions:
            logger.info("not plotting reconstructions")

    def plot_contribution_errors(self, contribution_errors: np.ndarray[np.ndarray], column_names):
        if self.draw_plots:
            logger.debug("plotting contribution errors")
            fig, ax = plt.subplots(figsize=(10, 6))

            for x in range(0, contribution_errors.shape[1]):  # for every channel
                ax.plot(contribution_errors[:, x], label=column_names[x])

            ax.set_title(f"Contribution errors")
            ax.set_xlabel("Timestamps")
            ax.set_ylabel("Absolute error")
            self.place_legend(ax)

            self.save_fig(fig, os.path.join(self.stats_path, "contribution_errors.png"), private_verbose=True)

    def plot_model_loss_val_loss(self, history):
        """plot model loss and val_loss"""

        if self.draw_plots:
            logger.debug("plotting loss and val_loss")
            fig = plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(2, 1, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title("Model metrics")
            ax1.plot(history.history["loss"], label="Training loss", color="blue")
            ax1.plot(history.history["val_loss"], label="Validation Loss", color="orange")
            ax1.set_ylabel("loss")

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(history.history["accuracy"], label="Training accuracy", color="purple")
            ax2.plot(history.history["val_accuracy"], label="Validation accuracy", color="olive")
            ax2.set_xlabel("epoch")
            ax2.set_ylabel("accuracy")

            fig.legend()

            self.save_fig(fig, os.path.join(self.stats_path, "model_metrics.png"))

    def plot_loss_histograms(self, train_loss: tf.Tensor, test_loss: tf.Tensor,
                             thresholds: tuple[float, float]):
        """histogram of loss values, with threshold"""

        if self.draw_plots:
            logger.debug("plotting loss histograms")
            max_loss = round(np.max(tf.concat([train_loss, test_loss], axis=0)), 2) + 0.01
            fig = plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(2, 1, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title("Reconstruction loss frequency in test_data")
            self.draw_loss_histogram(ax1, train_loss, thresholds, "Train", max_loss)

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set_xlabel("Reconstruction loss")
            self.draw_loss_histogram(ax2, test_loss, thresholds, "Test", max_loss)

            self.save_fig(fig, os.path.join(self.stats_path, "Loss_Histogram.png"))

    def draw_loss_histogram(self, ax: plt.Axes, loss, thresholds: tuple[float, float], title: str, max_loss: float):
        """Draws histogram with set styling for test and train loss"""

        ax.hist(loss, bins=50, label=f"{title} loss")
        ax.axvline(x=thresholds[0], color="r", label="threshold status 2", linestyle="dashed")
        ax.axvline(x=thresholds[1], color="r", label="threshold status 3")
        ax.set_xlim(0, max_loss)
        ax.set_ylabel("Frequency")
        self.place_legend(ax)

    def plot_loss_line_chart(self, title: str, loss: tf.Tensor, status_arr: list[int],
                             thresholds: tuple[float, float]):
        """
        plot of loss value for each test_data, with line for anomaly threshold
        parameter zoomed only changes title and file name
        """

        if self.draw_plots:
            design = 1
            if design == 0:
                # loss and status on same plot (needs to be better formatted)
                fig, ax = plt.subplots(figsize=(10, 6))
                self.draw_loss_line(ax, loss, thresholds, title)
                ax.plot(range(len(status_arr)), status_arr, label="status", color="orange")

                ax.set_ylim(0)
                ax.set_title(f"""reconstruction loss in {title}_data""")
                self.place_legend(ax)

            elif design == 1:
                # separate plots for loss and status
                fig = plt.figure(figsize=(10, 6))
                gs = gridspec.GridSpec(2, 1, figure=fig)

                ax1 = fig.add_subplot(gs[0, 0])
                self.draw_loss_line(ax1, loss, thresholds, title)

                ax1.set_ylim(0)
                ax1.set_title(f"""reconstruction loss in {title}_data""")
                self.place_legend(ax1)

                ax2 = fig.add_subplot(gs[1, 0])
                ax2.plot(range(len(status_arr)), status_arr, label="status", color="orange")
                self.place_legend(ax2)
            else:
                raise Exception("PlottingManager.plot_loss_line_chart() design not 0 or 1")

            self.save_fig(fig, os.path.join(self.stats_path, f"{title}_Loss.png"))

    def plot_zoomed_loss_line_chart(self, title: str, test_loss: tf.Tensor, thresholds: tuple[float, float]):
        """
        plot of loss value for test_data, zoomed into the largest loss, with line for anomaly threshold
        """

        if self.draw_plots:
            fig, ax = plt.subplots(figsize=(10, 6))

            y = test_loss
            self.draw_loss_line(ax, y, thresholds, title)

            max_x = range(len(y))[np.argmax(y)]  # x value of max_loss

            # padding of 50 indexes around max_x
            # max(max_x-50, 0) so the lowest index is not negative
            ax.set_xlim(max(max_x - 50, 0), max_x + 50)
            ax.set_ylim(0)
            ax.set_title(f"""reconstruction loss in {title}_data, zoomed to highest loss""")
            self.place_legend(ax)

            self.save_fig(fig, os.path.join(self.stats_path, f"{title}_Loss_Zoomed.png"))

    def draw_loss_line(self, ax: plt.Axes, y: tf.Tensor, thresholds: tuple[float, float], title: str):
        """draws line chart with set styling"""

        ax.plot(range(len(y)), y, label=f"{title} loss")
        ax.axhline(y=thresholds[0], color="r", label="threshold status 2", linestyle="dashed")
        ax.axhline(y=thresholds[1], color="r", label="threshold status 3")

        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Reconstruction loss")

    def save_fig(self, fig: plt.Figure, file_path: str, private_verbose: bool = True):
        """Saves pyplot fig to file_path & clears pyplot.
        verbose decides if file saved message displayed, default = True"""

        fig.savefig(file_path)
        plt.close()

        if private_verbose:
            logger.debug(f"fig saved to {file_path}")

    def clear_images_folder(self, folder: str):
        """clear folder of .png files"""
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            try:
                if (os.path.isfile(file_path) or os.path.islink(file_path)) and (".png" in file_name):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
