import os

import numpy as np
from utils.analyzer_utils.common_analyzer_utils import write_title_in_file
from utils.analyzer_utils.results_analyzer_utils import create_folds_directories_in_input_directory, write_analyzed_statistics_of_folds_in_file
from visualizer import VisualizerAnalyzer

from constants.base_directories_names_constants import RESULTS_DIRECTORY_NAME
from constants.common_constants import SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME
from utils.load_and_store_loss_and_scores import load_loss, load_scores

__all__ = ["ResultsAnalizerParent"]


class ResultsAnalizerParent:
    __REPORT_FILE_NAME = "analyze_results_report.txt"

    def __init__(self, analyze_directory, visualization_directory, logger):
        self.analyze_directory = analyze_directory
        self.analyze_report_file_path = self.analyze_directory / self.__REPORT_FILE_NAME
        self.visualization_directory = visualization_directory
        self._logger = logger

    def analyze_folds_results_gradient_descent_version(self, folds_outputs_directories):
        write_title_in_file("analyze folds results (gradient descent)", self.analyze_report_file_path)
        folds_paths_in_visualization_directory = create_folds_directories_in_input_directory(len(folds_outputs_directories), self.visualization_directory)
        folds_outputs_directories.sort()
        dummy_directory = folds_outputs_directories[0] / RESULTS_DIRECTORY_NAME
        if (dummy_directory / SPLIT_TRAIN_NAME).is_dir() and (dummy_directory / SPLIT_VALIDATION_NAME).is_dir():
            folds_train_scores_list_of_dict = []
            folds_validation_scores_list_of_dict = []
            folds_train_losses_list = []
            folds_validation_losses_list = []
            for fold_outputs_directory, fold_path in zip(folds_outputs_directories, folds_paths_in_visualization_directory):
                result_directory = fold_outputs_directory / RESULTS_DIRECTORY_NAME
                epochs_train_losses = load_loss(result_directory, SPLIT_TRAIN_NAME)
                epochs_validation_losses = load_loss(result_directory, SPLIT_VALIDATION_NAME)
                epochs_train_scores_list_of_dicts = load_scores(result_directory, SPLIT_TRAIN_NAME)
                epochs_validation_scores_list_of_dicts = load_scores(result_directory, SPLIT_VALIDATION_NAME)
                min_loss_validation_indice = np.argmin(epochs_validation_losses)
                VisualizerAnalyzer.visualize_split_train_and_validation_scores_and_loss_gradient_descent_version(
                    train_loss=epochs_train_losses,
                    validation_loss=epochs_validation_losses,
                    train_scores_list_of_dicts=epochs_train_scores_list_of_dicts,
                    validation_scores_list_of_dicts=epochs_validation_scores_list_of_dicts,
                    min_loss_validation_indice=min_loss_validation_indice,
                    directory_to_save_figure=fold_path,
                )

                folds_train_losses_list.append(epochs_train_losses[min_loss_validation_indice])
                folds_validation_losses_list.append(epochs_validation_losses[min_loss_validation_indice])
                folds_train_scores_list_of_dict.append(epochs_train_scores_list_of_dicts[min_loss_validation_indice])
                folds_validation_scores_list_of_dict.append(epochs_validation_scores_list_of_dicts[min_loss_validation_indice])

            folds_train_visualization_directory = self.visualization_directory / "folds" / SPLIT_TRAIN_NAME
            folds_validation_visualization_directory = self.visualization_directory / "folds" / SPLIT_VALIDATION_NAME
            os.makedirs(folds_train_visualization_directory, exist_ok=True)
            os.makedirs(folds_validation_visualization_directory, exist_ok=True)

            VisualizerAnalyzer.visualize_folds_metrics(
                split_name=SPLIT_TRAIN_NAME,
                loss_values_list=folds_train_losses_list,
                scores_list_of_dicts=folds_train_scores_list_of_dict,
                directory_to_save_figure=folds_train_visualization_directory,
            )
            VisualizerAnalyzer.visualize_folds_metrics(
                split_name=SPLIT_VALIDATION_NAME,
                loss_values_list=folds_validation_losses_list,
                scores_list_of_dicts=folds_validation_scores_list_of_dict,
                directory_to_save_figure=folds_validation_visualization_directory,
            )

            write_analyzed_statistics_of_folds_in_file(
                split_name=SPLIT_TRAIN_NAME,
                folds_losses_list=folds_train_losses_list,
                folds_scores_list_of_dict=folds_train_scores_list_of_dict,
                analyze_report_file_path=self.analyze_report_file_path,
            )
            write_analyzed_statistics_of_folds_in_file(
                split_name=SPLIT_VALIDATION_NAME,
                folds_losses_list=folds_validation_losses_list,
                folds_scores_list_of_dict=folds_validation_scores_list_of_dict,
                analyze_report_file_path=self.analyze_report_file_path,
            )

    def analyze_folds_results_machine_learning_version(self, folds_outputs_directories):
        write_title_in_file("analyze folds results (machine learning)", self.analyze_report_file_path)
        folds_outputs_directories.sort()
        dummy_directory = folds_outputs_directories[0] / RESULTS_DIRECTORY_NAME
        if (dummy_directory / SPLIT_VALIDATION_NAME).is_dir():
            folds_validation_scores_list_of_dict = []
            for fold_outputs_directory in folds_outputs_directories:
                result_directory = fold_outputs_directory / RESULTS_DIRECTORY_NAME
                epochs_scores_list_of_dicts = load_scores(result_directory, SPLIT_VALIDATION_NAME)
                folds_validation_scores_list_of_dict.append(epochs_scores_list_of_dicts)

            write_analyzed_statistics_of_folds_in_file(
                split_name=SPLIT_VALIDATION_NAME,
                folds_losses_list=None,
                folds_scores_list_of_dict=folds_validation_scores_list_of_dict,
                analyze_report_file_path=self.analyze_report_file_path,
            )

            folds_validation_visualization_directory = self.visualization_directory / "folds" / SPLIT_VALIDATION_NAME
            os.makedirs(folds_validation_visualization_directory, exist_ok=True)

            VisualizerAnalyzer.visualize_folds_metrics(
                split_name=SPLIT_VALIDATION_NAME,
                loss_values_list=None,
                scores_list_of_dicts=folds_validation_scores_list_of_dict,
                directory_to_save_figure=folds_validation_visualization_directory,
            )
