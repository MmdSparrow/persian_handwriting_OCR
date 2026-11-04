import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from utils.common_utils import folds_mean_std_calculator

__all__ = [
    "convert_encoded_label_to_str",
    "plot_channels_on_3d_brain",
    "plot_connectivity_on_3d_brain",
    "plot_single_split",
    "plot_train_and_validation_metric_diagram",
]


def convert_encoded_label_to_str(label):
    if label == 1:
        return "MDD"
    if label == 0:
        return "HC"
    raise ValueError()


def plot_train_and_validation_metric_diagram(
    metric_name,
    train_metric_values_list,
    validation_metric_values_list,
    marker_indice,
    train_diagram_color,
    validation_diagram_color,
    marker_color,
    marker_legends_label,
    directory_to_save_figure,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(train_metric_values_list) + 1)
    ax.plot(
        epochs,
        train_metric_values_list,
        label=f"Train {metric_name}",
        linewidth=2,
        color=train_diagram_color,
    )
    ax.plot(
        epochs,
        validation_metric_values_list,
        label=f"Validation {metric_name}",
        linewidth=2,
        color=validation_diagram_color,
    )
    epoch = epochs[marker_indice]
    marker_train_value = train_metric_values_list[marker_indice]
    marker_validation_value = validation_metric_values_list[marker_indice]
    ax.scatter(
        epoch,
        marker_train_value,
        color=marker_color,
        s=50,
        zorder=5,
    )
    ax.scatter(
        epoch,
        marker_validation_value,
        color=marker_color,
        s=50,
        zorder=5,
    )
    ax.annotate(
        f"{marker_train_value:.4f}",
        xy=(epoch, marker_train_value),
        xytext=(8, 8),
        textcoords="offset points",
        color=marker_color,
        fontsize=10,
    )
    ax.annotate(
        f"{marker_validation_value:.4f}",
        xy=(epoch, marker_validation_value),
        xytext=(8, -15),
        textcoords="offset points",
        color=marker_color,
        fontsize=10,
    )
    marker_legend = Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=marker_color,
        markeredgecolor=marker_color,
        markersize=7,
        label=marker_legends_label,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles=handles + [marker_legend],
        labels=labels + [marker_legends_label],
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.replace("_", " "))
    ax.set_title(f"Train vs Validation {metric_name.replace('_', ' ')}")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(directory_to_save_figure / f"train_and_validation_{metric_name.lower()}_curve.png")
    plt.close(fig)


def plot_single_split(
    split_name,
    metric_name,
    metric_values_list,
    marker_indice,
    diagram_color,
    marker_color,
    marker_legends_label,
    directory_to_save_visualizations,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(metric_values_list) + 1)
    ax.plot(
        epochs,
        metric_values_list,
        label=f"{split_name} {metric_name}",
        linewidth=2,
        color=diagram_color,
    )
    if marker_indice is not None:
        epoch = epochs[marker_indice]
        marker_value = metric_values_list[marker_indice]
        ax.scatter(
            epoch,
            marker_value,
            color=marker_color,
            s=50,
            zorder=5,
        )
        ax.annotate(
            f"{marker_value:.4f}",
            xy=(epoch, marker_value),
            xytext=(8, 8),
            textcoords="offset points",
            color=marker_color,
            fontsize=10,
        )
        marker_legend = Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=marker_color,
            markeredgecolor=marker_color,
            markersize=7,
            label=marker_legends_label,
        )
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles + [marker_legend],
            labels=labels + [marker_legends_label],
        )
    else:
        fig.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name.replace("_", " "))
    ax.set_title(f"{split_name} {metric_name.replace('_', ' ')}")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(directory_to_save_visualizations / f"{metric_name.lower()}_curve.png")
    plt.close(fig)


def plot_channels_on_3d_brain(
    channels_names,
    montage_name,
):
    brain = mne.viz.Brain(
        "fsaverage",
        alpha=1.0,
        title="test title",
        hemi="both",
        surf="pial",
        background="white",
        cortex="low_contrast",
        size=(1000, 800),
        show=False,
    )

    brain.show_view("dorsal")
    image = brain.screenshot()
    brain.close()

    montage = mne.channels.make_standard_montage(montage_name)
    positions = montage.get_positions()["ch_pos"]

    xy = []
    for ch_name in channels_names:
        if ch_name in positions:
            xyz = positions[ch_name]
            xy.append([xyz[0], xyz[1]])
    xy = np.asarray(xy)

    # normalize to image coordiantes
    x = xy[:, 0]
    y = xy[:, 1]
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    x_margin = 0.20
    y_margin = 0.10
    x_norm = x_margin + x_norm * (1 - 2 * x_margin)
    y_norm = y_margin + y_norm * (1 - 2 * y_margin)
    height, width = image.shape[:2]
    x_pixels = x_norm * width
    # matplotlib's origin is bottom-left
    y_pixels = (1 - y_norm) * height
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    for ch_name, x_pixel, y_pixel in zip(
        channels_names,
        x_pixels,
        y_pixels,
    ):
        ax.scatter(
            x_pixels,
            y_pixels,
            s=18,
            c="black",
            marker="o",
            linewidths=0.5,
            zorder=10,
        )
        ax.text(
            x_pixel,
            y_pixel - 5,
            ch_name,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="black",
            zorder=11,
        )

    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig, ax


def plot_connectivity_on_3d_brain(
    connectivity_matrix,
    channels_names,
    montage_name,
    color,
    value_threshold,
    min_linewidth=0.5,
    max_linewidth=6.0,
    title="Connectivity",
):
    brain = mne.viz.Brain(
        "fsaverage",
        alpha=1.0,
        title=title,
        hemi="both",
        surf="pial",
        background="white",
        cortex="low_contrast",
        size=(1000, 800),
        show=False,
    )
    brain.show_view("dorsal")
    image = brain.screenshot()
    brain.close()

    montage = mne.channels.make_standard_montage(montage_name)
    positions = montage.get_positions()["ch_pos"]

    xy = []
    valid_channels = []
    for ch_name in channels_names:
        if ch_name in positions:
            xyz = positions[ch_name]
            xy.append([xyz[0], xyz[1]])
            valid_channels.append(ch_name)
    xy = np.asarray(xy)

    x = xy[:, 0]
    y = xy[:, 1]
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    x_margin = 0.23
    y_margin = 0.10
    x_norm = x_margin + x_norm * (1 - 2 * x_margin)
    y_norm = y_margin + y_norm * (1 - 2 * y_margin)
    height, width = image.shape[:2]
    x_pixels = x_norm * width
    # matplotlib's origin is bottom-left
    y_pixels = (1 - y_norm) * height
    channel_to_pixel = {ch_name: (x_pixels[idx], y_pixels[idx]) for idx, ch_name in enumerate(valid_channels)}

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)

    connectivity_values_abs = np.abs(connectivity_matrix)
    for ch_i_idx in range(len(channels_names)):
        for ch_j_idx in range(ch_i_idx + 1, len(channels_names)):
            ch_i = channels_names[ch_i_idx]
            ch_j = channels_names[ch_j_idx]
            abs_connectivity_value = connectivity_values_abs[ch_i_idx, ch_j_idx]
            if abs_connectivity_value > value_threshold:
                linewidth = min_linewidth + abs_connectivity_value * (max_linewidth - min_linewidth)
                x2, y2 = channel_to_pixel[ch_j]
                x1, y1 = channel_to_pixel[ch_i]
                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color=color,
                    linewidth=linewidth,
                    alpha=0.5,
                    zorder=5,
                )

    for ch_name in valid_channels:
        x_pixel, y_pixel = channel_to_pixel[ch_name]
        ax.scatter(
            x_pixel,
            y_pixel,
            s=18,
            c="black",
            marker="o",
            linewidths=0.5,
            zorder=10,
        )
        ax.text(
            x_pixel,
            y_pixel - 5,
            ch_name,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="black",
            zorder=11,
        )

    ax.text(
        0.98,
        0.98,
        (f"Threshold Before Max Normalization: {value_threshold:.4f}"),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.9,
        },
        zorder=20,
    )
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig, ax


def plot_metric_across_folds(
    metric_name,
    metric_values,
    diagram_color,
    hline_color,
    directory_to_save_figure,
):
    metric_values = np.asarray(metric_values)
    folds = np.arange(len(metric_values))
    mean_value, _ = folds_mean_std_calculator(metric_values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        folds,
        metric_values,
        label=metric_name,
        color=diagram_color,
    )
    ax.vlines(
        x=folds,
        ymin=0,
        ymax=metric_values,
        color=diagram_color,
        linewidth=1.2,
    )
    ax.axhline(
        y=mean_value,
        color=hline_color,
        linestyle="--",
        label=f"Mean = {mean_value:.4f}",
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} Across Folds")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=30, integer=True))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(directory_to_save_figure / f"{metric_name.lower()}_folds.png")
    plt.close(fig)
