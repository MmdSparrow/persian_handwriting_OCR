import shutil
from abc import ABC, abstractmethod

import numpy as np

from constants.base_directories_names_constants import DATASET_IMAGES_DIRECTORY_NAME
from utils.load_and_store_dataset_labels_utils import generate_data_labels_csv_file


class InitialPreparationParent(ABC):
    def __init__(self, dataset_directory):
        super().__init__()
        self._dataset_directory = dataset_directory

    @abstractmethod
    def _list_data_paths_and_labels(self):
        pass

    @abstractmethod
    def _load_data(self, file_path):
        pass

    def restruct_dataset(self):
        data_paths_list, labels_list = self._list_data_paths_and_labels()
        destination_dir = self._dataset_directory / DATASET_IMAGES_DIRECTORY_NAME
        destination_dir.mkdir(parents=True, exist_ok=True)
        data_files_names_list = []
        for file_path in data_paths_list:
            data_files_names_list.append(file_path.stem)
            if file_path.format == "npy":
                shutil.move(file_path, destination_dir)
            else:
                image = self._load_data(file_path)
                np.save((destination_dir / file_path.stem).with_suffix(".npy"), image)
        generate_data_labels_csv_file(data_files_names_list, labels_list, self._dataset_directory)
