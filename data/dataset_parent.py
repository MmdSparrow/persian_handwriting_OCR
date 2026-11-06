from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.load_and_store_dataset_labels_utils import get_data_label, read_csv_file
from utils.load_and_store_dataset_metadata import load_dataset_metadata

__all__ = ["DatasetParent"]


class DatasetParent(Dataset):
    def __init__(
        self,
        root,
        data_paths_list,
        augmentation=None,
    ):
        self.__chosen_preprocesses_list, self.__preprocesses_hyperparameters_kwargs = load_dataset_metadata(root)
        self.df_labels = read_csv_file(root)
        self.augmentation = augmentation
        self.samples_paths = [Path(data_file) for data_file in sorted(data_paths_list)]

    def get_all_metadata(self):
        return self.__chosen_preprocesses_list, self.__preprocesses_hyperparameters_kwargs

    def __len__(self):
        return len(self.samples_paths)

    def __getitem__(self, idx):
        sample_path = self.samples_paths[idx]
        data = np.load(sample_path)
        data = torch.from_numpy(data)
        label = get_data_label(sample_path, self.df_labels)
        if self.augmentation is not None:
            data = self.augmentation(data)
        return data.float(), label
