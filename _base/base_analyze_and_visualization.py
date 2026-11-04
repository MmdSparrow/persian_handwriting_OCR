import os

from constants.base_directories_names_constants import (
    ANALYZE_DIRECTORY_NAME,
    RESULTS_DIRECTORY_NAME,
    SPLITED_DATA_DIRECTORY_NAME,
    VISUALIZATIONS_DIRECTORY_NAME,
)

from ._base_parent import BaseParent

__all__ = ["BaseAnalyzeAndVisualization"]


class BaseAnalyzeAndVisualization(BaseParent):
    def __init__(self, verbose=True, output_directory=None):
        super().__init__(verbose, output_directory)

    def __create_visualizations_directory(self):
        self._results_directory = self._output_directory / RESULTS_DIRECTORY_NAME
        self._visulizations_directory = self._results_directory / VISUALIZATIONS_DIRECTORY_NAME
        os.makedirs(self._visulizations_directory, exist_ok=True)

    def __create_analyzes_directory(self):
        self._analyzes_directory = self._output_directory / ANALYZE_DIRECTORY_NAME
        os.makedirs(self._analyzes_directory, exist_ok=True)

    def __create_emtpy_split_directory(self, dataset_root_directory):
        self._splited_data_directory = dataset_root_directory / SPLITED_DATA_DIRECTORY_NAME
        os.makedirs(self._splited_data_directory, exist_ok=True)

    def create_required_analyze_raw_and_preprocessed_data_directory(self, dataset_root_directory):
        self.__create_analyzes_directory()
        self.__create_emtpy_split_directory(dataset_root_directory)

    def create_required_analyze_feature_extracted_data_directory(self):
        self.__create_analyzes_directory()

    def create_required_analyze_results_directories(self):
        self.__create_analyzes_directory()
        self.__create_visualizations_directory()
