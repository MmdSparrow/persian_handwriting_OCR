from abc import ABC, abstractmethod

from utils.load_and_store_dataset_labels_utils import generate_data_labels_csv_file


class InitialPreparationParent(ABC):
    def __init__(self, dataset_directory):
        super().__init__()
        self._dataset_directory = dataset_directory

    @abstractmethod
    def _list_data_names_and_labels(self):
        pass

    def generate_data_labels(self):
        data_files_names_list, labels_list = self._list_data_names_and_labels()
        generate_data_labels_csv_file(data_files_names_list, labels_list, self._dataset_directory)
