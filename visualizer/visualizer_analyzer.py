from constants.common_constants import SPLIT_TEST_NAME, SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME
from utils.visualizer_utils import plot_metric_across_folds, plot_train_and_validation_metric_diagram

from .visualizer_parent import VisualizerParent


class VisualizerAnalyzer(VisualizerParent):
    def __init__(self):
        super().__init__()

    @staticmethod
    def visualize_split_train_and_validation_scores_and_loss_gradient_descent_version(
        train_loss,
        validation_loss,
        train_scores_list_of_dicts,
        validation_scores_list_of_dicts,
        min_loss_validation_indice,
        directory_to_save_figure,
    ):
        plot_train_and_validation_metric_diagram(
            metric_name="Loss",
            train_metric_values_list=train_loss,
            validation_metric_values_list=validation_loss,
            marker_indice=min_loss_validation_indice,
            train_diagram_color=VisualizerParent._TRAIN_DIAGRAM_COLOR,
            validation_diagram_color=VisualizerParent._VALIDATION_DIAGRAM_COLOR,
            marker_color=VisualizerParent._ADDITIONAL_INFO_COLOR,
            marker_legends_label=VisualizerParent._MARKER_LEGENDS_LABEL,
            directory_to_save_figure=directory_to_save_figure,
        )
        for metric_name in train_scores_list_of_dicts[0]:
            train_metric_values_list = [score[metric_name] for score in train_scores_list_of_dicts]
            validation_metric_values_list = [score[metric_name] for score in validation_scores_list_of_dicts]
            plot_train_and_validation_metric_diagram(
                metric_name=metric_name,
                train_metric_values_list=train_metric_values_list,
                validation_metric_values_list=validation_metric_values_list,
                marker_indice=min_loss_validation_indice,
                train_diagram_color=VisualizerParent._TRAIN_DIAGRAM_COLOR,
                validation_diagram_color=VisualizerParent._VALIDATION_DIAGRAM_COLOR,
                marker_color=VisualizerParent._ADDITIONAL_INFO_COLOR,
                marker_legends_label=VisualizerParent._MARKER_LEGENDS_LABEL,
                directory_to_save_figure=directory_to_save_figure,
            )

    @staticmethod
    def visualize_folds_metrics(
        split_name,
        loss_values_list,
        scores_list_of_dicts,
        directory_to_save_figure,
    ):
        diagram_color = None
        if split_name == SPLIT_TRAIN_NAME:
            diagram_color = VisualizerParent._TRAIN_DIAGRAM_COLOR
        elif split_name == SPLIT_VALIDATION_NAME:
            diagram_color = VisualizerParent._VALIDATION_DIAGRAM_COLOR
        elif split_name == SPLIT_TEST_NAME:
            diagram_color = VisualizerParent._TEST_DIAGRAM_COLOR
        if loss_values_list is not None:
            plot_metric_across_folds(
                metric_name="Loss",
                metric_values=loss_values_list,
                diagram_color=diagram_color,
                hline_color=VisualizerParent._ADDITIONAL_INFO_COLOR,
                directory_to_save_figure=directory_to_save_figure,
            )
        for metric_name in scores_list_of_dicts[0]:
            metric_values_list = [score[metric_name] for score in scores_list_of_dicts]
            plot_metric_across_folds(
                metric_name=metric_name,
                metric_values=metric_values_list,
                diagram_color=diagram_color,
                hline_color=VisualizerParent._ADDITIONAL_INFO_COLOR,
                directory_to_save_figure=directory_to_save_figure,
            )
