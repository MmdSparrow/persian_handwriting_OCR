import torch
from torch.utils.data import Dataset

from utils.data_utils import padding_labels
from utils.load_and_store_dataset_info import load_dataset_info
from utils.load_and_store_dataset_labels_utils import read_csv_file
from utils.load_and_store_image_utils import image_loader_selector
from utils.preprocess_utils.preprocess_function import apply_preprocesses

__all__ = ["DatasetParent"]


class DatasetParent(Dataset):
    def __init__(
        self,
        dataset_directory,
        data_files_names_list,
        dataset_specific_preprocesses,
        chosen_common_preprocesses_list,
        chosen_common_preprocesses_hyperparameters_dict,
        augmentation,
        seed,
    ):
        self._seed = seed
        images_format, images_directory, maximum_label_length = load_dataset_info(dataset_directory)
        self.__load_image_func = image_loader_selector(images_format)
        self.__dataset_specific_preprocesses = dataset_specific_preprocesses
        self.__chosen_common_preprocesses_list = chosen_common_preprocesses_list
        self.__chosen_common_preprocesses_hyperparameters_dict = chosen_common_preprocesses_hyperparameters_dict
        data_files_names_list, labels_list = read_csv_file(dataset_directory)
        # TODO: ADD SOS or EOS IF NEEDED.....
        labels_lengths = [len(label) for label in labels_list]
        self.augmentation = augmentation
        # [([sample_path], label) for sample_path, label in zip(samples_paths, samples_labels)]
        self.samples = [
            (sample_path, label, label_lenggth)
            for sample_path, label, label_lenggth in zip(
                [f"{dataset_directory / images_directory / data_file_name}{images_format}" for data_file_name in data_files_names_list],
                padding_labels(labels_list, maximum_label_length),
                labels_lengths,
            )
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, sample_label, sample_label_length = self.samples[idx]
        data = self.__load_image_func(sample_path)
        data = self.__dataset_specific_preprocesses.apply_specific_prepreocesses(data)
        data = apply_preprocesses(
            image=data,
            chosen_common_preprocesses_list=self.__chosen_common_preprocesses_list,
            preprocesses_hyperparameters_kwargs=self.__chosen_common_preprocesses_hyperparameters_dict,
            seed=self._seed,
        )
        data = torch.from_numpy(data)
        if self.augmentation is not None:
            data = self.augmentation(data)
        return data.float(), sample_label, sample_label_length
