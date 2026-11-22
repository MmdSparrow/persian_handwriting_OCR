from abc import ABC, abstractmethod

from utils.load_and_store_dataset_info import store_dataset_info
from utils.load_and_store_dataset_labels_utils import generate_data_labels_csv_file


class InitialPreparationParent(ABC):
    __IMAGES_FORMAT = None
    __IMAGES_DIRECTORY = None
    __MAXIMUM_LABEL_LENGTH= None

    def __init__(self, dataset_directory, logger):
        super().__init__()
        self._dataset_directory = dataset_directory
        self._logger = logger

    def _log_dataset_info(self):
        self._logger.info(f"image format: {self.__IMAGES_FORMAT}")
        self._logger.info(f"image directory: {self.__IMAGES_DIRECTORY}")

    @abstractmethod
    def _list_data_names_and_labels(self):
        pass

    def _generate_data_labels(self):
        data_files_names_list, labels_list = self._list_data_names_and_labels()
        generate_data_labels_csv_file(data_files_names_list, labels_list, self._dataset_directory)

    def initial_preparation(self):
        self._log_dataset_info()
        store_dataset_info(
            images_format=self.__IMAGES_FORMAT,
            images_directory=self.__IMAGES_DIRECTORY,
            maximum_label_length=self.__MAXIMUM_LABEL_LENGTH,
            directories_to_store_info_file=self._dataset_directory,
        )
        self._generate_data_labels()
