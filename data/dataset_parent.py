import numpy as np
import torch
from torch.utils.data import Dataset

from utils.load_and_store_dataset_labels_utils import get_data_labels, read_csv_file
from utils.load_and_store_dataset_preprocess_info import store_dataset_preprocesses_info
from utils.preprocess_utils.preprocess_function import apply_preprocesses

__all__ = ["DatasetParent"]


class DatasetParent(Dataset):
    _IMAGE_DIRECTORY = None
    _IMAGE_FORMAT = None

    def __init__(
        self,
        dataset_directory,
        data_files_names_list,
        chosen_preprocesses_list,
        chosen_preprocesses_hyperparameters_dict,
        augmentation=None,
    ):
        self.__chosen_preprocesses_list = chosen_preprocesses_list
        self.__chosen_preprocesses_hyperparameters_dict = chosen_preprocesses_hyperparameters_dict
        store_dataset_preprocesses_info(
            chosen_preprocesses_list,
            chosen_preprocesses_hyperparameters_dict,
            dataset_directory,
        )
        df_labels = read_csv_file(dataset_directory)
        self.augmentation = augmentation
        self.samples = zip(
            [f"{dataset_directory / self._IMAGE_DIRECTORY / data_file_name}{self._IMAGE_FORMAT}" for data_file_name in data_files_names_list],
            get_data_labels(data_files_names_list, df_labels),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, sample_label = self.samples[idx]
        data = np.load(sample_path)
        # TODO: dataset specific preprocess like remove background
        data = apply_preprocesses(
            image=data,
            chosen_common_preprocesses_list=self.__chosen_preprocesses_list,
            preprocesses_hyperparameters_kwargs=self.__chosen_preprocesses_hyperparameters_dict,
            seed=self._seed,
        )
        data = torch.from_numpy(data)
        if self.augmentation is not None:
            data = self.augmentation(data)
        return data.float(), sample_label
