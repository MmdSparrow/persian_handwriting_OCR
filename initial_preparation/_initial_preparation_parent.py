from abc import ABC, abstractmethod

from utils.load_and_store_dataset_info import generate_dataset_info_file
from utils.load_and_store_dataset_labels_utils import generate_data_labels_csv_file


class InitialPreparationParent(ABC):
    _IMAGES_FORMAT = None
    _IMAGES_DIRECTORY = None
    _MAXIMUM_LABEL_LENGTH= None

    def __init__(self, dataset_directory, logger):
        super().__init__()
        self._dataset_directory = dataset_directory
        self._logger = logger

    def _log_dataset_info(self):
        self._logger.info(f"image format: {self._IMAGES_FORMAT}")
        self._logger.info(f"image directory: {self._IMAGES_DIRECTORY}")

    @abstractmethod
    def _list_data_names_and_labels(self):
        pass

    def _generate_data_labels(self):
        data_files_names_list, labels_list = self._list_data_names_and_labels()
        generate_data_labels_csv_file(data_files_names_list, labels_list, self._dataset_directory)

    def initial_preparation(self):
        self._log_dataset_info()
        generate_dataset_info_file(
            images_format=self._IMAGES_FORMAT,
            images_directory=self._IMAGES_DIRECTORY,
            maximum_label_length=self._MAXIMUM_LABEL_LENGTH,
            directories_to_store_info_file=[self._dataset_directory],
        )
        self._generate_data_labels()
