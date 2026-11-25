import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from utils.common_utils import folds_mean_std_calculator

__all__ = [
    "plot_single_split",
    "plot_train_and_validation_metric_diagram",
]


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
