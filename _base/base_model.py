import os

from constants.base_directories_names_constants import (
    CHECKPOINT_DIRECTORY_NAME,
    RESULTS_DIRECTORY_NAME,
    SAMPLES_DIRECTORY_NAME,
)
from constants.common_constants import SPLIT_TEST_NAME, SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME

from ._base_parent import BaseParent

__all__ = ["BaseModel"]


class BaseModel(BaseParent):
    def __init__(self, outputs_directory, verbose=True):
        super().__init__(verbose, outputs_directory)

    def __create_results_directory(self, split_name):
        self._results_directory = self._outputs_directory / RESULTS_DIRECTORY_NAME
        os.makedirs(self._results_directory / split_name, exist_ok=True)

    def __create_samples_directory(self, split_name):
        self._samples_directory = self._outputs_directory / SAMPLES_DIRECTORY_NAME
        os.makedirs(self._samples_directory / split_name, exist_ok=True)

    def __create_checkpoint_directory(self):
        self._checkpoint_directory = self._outputs_directory / CHECKPOINT_DIRECTORY_NAME
        os.makedirs(self._checkpoint_directory, exist_ok=True)

    def create_train_neural_network_required_directories(self):
        self.__create_results_directory(SPLIT_TRAIN_NAME)
        self.__create_results_directory(SPLIT_VALIDATION_NAME)
        self.__create_checkpoint_directory()
        self.__create_samples_directory(SPLIT_TRAIN_NAME)
        self.__create_samples_directory(SPLIT_VALIDATION_NAME)

    def create_test_required_directories(self):
        self.__create_results_directory(SPLIT_TEST_NAME)
        self.__create_samples_directory(SPLIT_TEST_NAME)
