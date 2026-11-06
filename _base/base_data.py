import os
from pathlib import Path

from constants.base_directories_names_constants import PROCESSED_DATA_DIRECTORY_NAME
from constants.common_constants import SPLITS_NAMES

from ._base_parent import BaseParent

__all__ = ["BaseData"]


class BaseData(BaseParent):
    def __init__(self, dataset_directory, verbose=True, outputs_directory=None, seed=None):
        super().__init__(verbose, outputs_directory)
        self._dataset_directory = Path(dataset_directory)
        self._seed = seed

    def __create_preprocessed_data_directory(self, pipeline):
        self._preprocessed_data_directory = self._dataset_directory / PROCESSED_DATA_DIRECTORY_NAME
        for split_name in SPLITS_NAMES:
            os.makedirs(self._preprocessed_data_directory / pipeline / split_name, exist_ok=True)

    def create_preprocessed_data_required_directories(self, pipeline):
        self.__create_preprocessed_data_directory(pipeline)
