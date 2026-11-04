import shutil
from abc import ABC, abstractmethod

from constants.base_directories_names_constants import DATASET_IMAGES_DIRECTORY_NAME
from utils.load_and_store_dataset_labels_utils import generate_data_labels_csv_file


class InitialPreparationParent(ABC):
    _DATASET_NAME = None

    def __init__(self, dataset_root_directory):
        super().__init__()
        self._dataset_directory = dataset_root_directory

    @abstractmethod
    def _list_data_paths_and_labels(self):
        pass

    def restruct_dataset(self):
        data_paths_list, labels_list = self._list_data_paths_and_labels()
        destination_dir = self._dataset_directory / self._DATASET_NAME / DATASET_IMAGES_DIRECTORY_NAME
        destination_dir.mkdir(parents=True, exist_ok=True)
        data_files_names_list = []
        for file_path in data_paths_list:
            data_file_name = file_path.name
            shutil.move(file_path, destination_dir / data_file_name)
            data_files_names_list.append(data_file_name)
        generate_data_labels_csv_file(data_files_names_list, labels_list, self._dataset_directory / self._DATASET_NAME)
