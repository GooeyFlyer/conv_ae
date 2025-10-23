import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
import os


class PlottingManager:
    def __init__(self, draw_plots: bool, num_to_show: int):
        self.num_to_show = num_to_show
        self.draw_plots = draw_plots
        self.plots_path = "images/plots"
        self.stats_path = "images/stats"

        if self.draw_plots:
            self.clear_images_folder(self.plots_path)
            self.clear_images_folder(self.stats_path)

    def plot_reconstructions(self, test_data, decoded_data, column_names):
        """plot test data against reconstructed test data"""

        if self.draw_plots:
            print("plotting test data against reconstructed data")
            print(f"only first {self.num_to_show} datapoints")
            for x in range(0, len(test_data[0])):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot((test_data[:, x])[:self.num_to_show], label=f"test {column_names[x]}", color="b")
                ax.plot((decoded_data[:, x])[:self.num_to_show], label=f"reconstructed {column_names[x]}", color="r")
                ax.fill_between(
                    np.arange(len((test_data[:, x])[:self.num_to_show])),
                    (decoded_data[:, x])[:self.num_to_show],
                    (test_data[:, x])[:self.num_to_show],
                    label="error", color="lightcoral"
                )

                # ax.set_xticklabels(date_time_series)
                ax.set_ylim(0, 1)
                ax.set_title(f"Plots of {column_names[x]}")
                ax.set_xlabel("Timestamps")
                ax.set_ylabel("Normalised values")
                ax.legend()

                file_name = f"plot_{column_names[x]}.png"
                self.save_fig(fig, os.path.join(self.plots_path, file_name))

            print("plots saved to images/plots/")

    def plot_model_loss_val_loss(self, history):
        """plot model loss and val_loss"""

        if self.draw_plots:
            print("plotting loss and val_loss")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history.history["loss"], label="Training loss")
            ax.plot(history.history["val_loss"], label="Validation Loss")
            ax.legend()

            self.save_fig(fig, os.path.join(self.stats_path, "loss-val_loss.png"))

    def plot_loss_histograms(self, train_loss: tf.Tensor, test_loss: tf.Tensor, threshold: float):
        """histogram of loss values, with threshold"""

        if self.draw_plots:
            print("plotting loss histograms")
            max_loss = round(np.max(tf.concat([train_loss, test_loss], axis=0)), 2) + 0.01
            fig = plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(2, 1, figure=fig)

            ax1 = fig.add_subplot(gs[0, 0])
            ax1.set_title("Reconstruction loss frequency in test_data")
            self.draw_loss_histogram(ax1, train_loss, threshold, "Train", max_loss)

            ax2 = fig.add_subplot(gs[1, 0])
            ax2.set_xlabel("Reconstruction loss")
            self.draw_loss_histogram(ax2, test_loss, threshold, "Test", max_loss)

            self.save_fig(fig, os.path.join(self.stats_path, "Loss_Histogram.png"))

    def draw_loss_histogram(self, ax: plt.Axes, loss, threshold: float, title: str, max_loss: float):
        """Draws histogram with set styling for test and train loss"""

        ax.hist(loss, bins=50, label=f"{title} loss")
        ax.axvline(x=threshold, color="r", label="anomaly threshold")
        ax.set_xlim(0, max_loss)
        ax.set_ylabel("Frequency")
        ax.legend()

    def plot_loss_bar_chart(self, test_loss: tf.Tensor, threshold: float):
        """bar chart of loss value for each test_data, with bar for anomaly threshold"""

        if self.draw_plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(test_loss[:self.num_to_show])), test_loss[:self.num_to_show], label=f"test loss")
            ax.axhline(y=threshold, color="r", label="anomaly threshold")
            ax.set_title(f"reconstruction loss in first {self.num_to_show} timestamps in test_data")
            ax.set_xlabel("Timestamp")
            ax.set_ylabel("Reconstruction loss")
            ax.legend()

            self.save_fig(fig, os.path.join(self.stats_path, "Test_Loss.png"))

    def save_fig(self, fig: plt.Figure, file_path: str):
        fig.savefig(file_path)
        plt.close()
        print("fig saved to ", file_path)

    def clear_images_folder(self, folder: str):
        """clear folder of .png files"""
        for file_name in os.listdir(folder):
            file_path = os.path.join(folder, file_name)
            try:
                if (os.path.isfile(file_path) or os.path.islink(file_path)) and (".png" in file_name):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
