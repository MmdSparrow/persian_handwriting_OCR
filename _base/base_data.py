import os
import shutil
from pathlib import Path

from constants.base_directories_names_constants import (
    PROCESSED_DATA_DIRECTORY_NAME,
    SPLITED_DATA_DIRECTORY_NAME,
)
from constants.common_constants import SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME, SPLITS_NAMES

from ._base_parent import BaseParent

__all__ = ["BaseData"]


class BaseData(BaseParent):
    def __init__(self, dataset_root_directory, verbose=True, output_directory=None, seed=None):
        super().__init__(verbose, output_directory)
        self._dataset_directory = Path(dataset_root_directory)
        self._seed = seed

    def recreate_train_and_validation_splits_and_preprocess_directories(self, pipeline):
        self._splited_data_directory = self._dataset_directory / SPLITED_DATA_DIRECTORY_NAME
        self._preprocessed_data_directory = self._dataset_directory / PROCESSED_DATA_DIRECTORY_NAME
        shutil.rmtree(self._splited_data_directory / SPLIT_TRAIN_NAME, ignore_errors=True)
        shutil.rmtree(self._splited_data_directory / SPLIT_VALIDATION_NAME, ignore_errors=True)
        shutil.rmtree(self._splited_data_directory / pipeline / SPLIT_TRAIN_NAME, ignore_errors=True)
        shutil.rmtree(self._splited_data_directory / pipeline / SPLIT_VALIDATION_NAME, ignore_errors=True)
        shutil.rmtree(self._preprocessed_data_directory / pipeline / SPLIT_TRAIN_NAME)
        shutil.rmtree(self._preprocessed_data_directory / pipeline / SPLIT_VALIDATION_NAME)
        self.__create_preprocessed_data_directory(pipeline)

    def __create_emtpy_split_directory(self):
        self._splited_data_directory = self._dataset_directory / SPLITED_DATA_DIRECTORY_NAME
        os.makedirs(self._splited_data_directory, exist_ok=True)

    def __create_preprocessed_data_directory(self, pipeline):
        self._preprocessed_data_directory = self._dataset_directory / PROCESSED_DATA_DIRECTORY_NAME
        for split_name in SPLITS_NAMES:
            os.makedirs(self._preprocessed_data_directory / pipeline / split_name, exist_ok=True)

    def create_prepared_data_required_data_directories(self):
        self.__create_emtpy_split_directory()

    def create_splits_directories(self):
        self._splited_data_directory = self._dataset_directory / SPLITED_DATA_DIRECTORY_NAME
        for split_name in SPLITS_NAMES:
            os.makedirs(self._splited_data_directory / split_name, exist_ok=True)

    def create_preprocessed_data_required_directories(self, pipeline):
        self._splited_data_directory = self._dataset_directory / SPLITED_DATA_DIRECTORY_NAME
        self.__create_preprocessed_data_directory(pipeline)
