import cv2
import numpy as np

from constants.base_directories_names_constants import DATASET_IMAGES_DIRECTORY_NAME
from utils.load_and_store_dataset_labels_utils import read_csv_file

from ._initial_preparation_parent import InitialPreparationParent


class InitialPreparationTypedWrdsImgesDatasets(InitialPreparationParent):
    def __init__(self, dataset_directory):
        super().__init__(dataset_directory)

    def _list_data_paths_and_labels(self):
        self.df_labels = read_csv_file(self._dataset_directory)
        data_directory = self._dataset_directory / DATASET_IMAGES_DIRECTORY_NAME
        data_list = self.df_labels["Data Name"].tolist()
        labels_list = self.df_labels["Label"].tolist()
        return [data_directory / (str(data_name) + ".tif") for data_name in data_list], labels_list

    def _load_data(self, file_path):
        return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    def restruct_dataset(self):
        data_paths_list, _ = self._list_data_paths_and_labels()
        destination_dir = self._dataset_directory / DATASET_IMAGES_DIRECTORY_NAME
        destination_dir.mkdir(parents=True, exist_ok=True)
        data_files_names_list = []
        for file_path in data_paths_list:
            data_files_names_list.append(file_path.stem)
            image = self._load_data(file_path)
            np.save((destination_dir / file_path.stem).with_suffix(".npy"), image)
