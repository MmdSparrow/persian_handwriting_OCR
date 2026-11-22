import torch
from torch.utils.data import Dataset

from constants.common_constants import CHOSEN_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT, CHOSEN_PREPROCESSES_LIST_IN_ORDER_OF_APPLY
from utils.data_utils import padding_labels
from utils.load_and_store_dataset_info import load_dataset_info
from utils.load_and_store_dataset_labels_utils import get_data_labels, read_csv_file
from utils.load_and_store_dataset_preprocess_info import store_dataset_preprocesses_info
from utils.load_and_store_image_utils import image_loader_selector
from utils.preprocess_utils.preprocess_function import apply_preprocesses

__all__ = ["DatasetParent"]


class DatasetParent(Dataset):
    def __init__(
        self,
        dataset_directory,
        data_files_names_list,
        dataset_specific_preprocesses,
        chosen_dataset_specific_preprocesses_name_list,
        chosen_dataset_specific_preprocesses_hyperparameters,
        chosen_common_preprocesses_list,
        chosen_common_preprocesses_hyperparameters_dict,
        augmentation,
    ):
        self._log_chosen_preprocesses_info(
            chosen_preprocesses_name_list=chosen_dataset_specific_preprocesses_name_list,
            chosen_preprocesses_hyperparameters_dict=chosen_dataset_specific_preprocesses_hyperparameters,
            preprocesses_name_list_in_order_of_apply=dataset_specific_preprocesses._CHOSEN_DATASET_SPECIFIC_PREPROCESSES_NAME_LIST_IN_ORDER_OF_APPLY,
            preprocesses_and_their_hyperparameters_dict=dataset_specific_preprocesses._CHOSEN_DATASET_SPECIFIC_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT,
            preprocesses_type="dataset-specific",
        )
        self._log_chosen_preprocesses_info(
            chosen_preprocesses_name_list=chosen_common_preprocesses_list,
            chosen_preprocesses_hyperparameters_dict=chosen_common_preprocesses_hyperparameters_dict,
            preprocesses_name_list_in_order_of_apply=CHOSEN_PREPROCESSES_LIST_IN_ORDER_OF_APPLY,
            preprocesses_and_their_hyperparameters_dict=CHOSEN_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT,
            preprocesses_type="common",
        )
        images_format, images_directory, maximum_label_length = load_dataset_info(dataset_directory)
        self.__load_image_func = image_loader_selector(images_format)
        self.__dataset_specific_preprocesses = dataset_specific_preprocesses
        self.__chosen_dataset_specific_preprocesses_name_list = chosen_dataset_specific_preprocesses_name_list
        self.__chosen_dataset_specific_preprocesses_hyperparameters = chosen_dataset_specific_preprocesses_hyperparameters
        self.__chosen_common_preprocesses_list = chosen_common_preprocesses_list
        self.__chosen_common_preprocesses_hyperparameters_dict = chosen_common_preprocesses_hyperparameters_dict
        store_dataset_preprocesses_info(
            chosen_common_preprocesses_list,
            chosen_common_preprocesses_hyperparameters_dict,
            dataset_directory,
        )
        df_labels = read_csv_file(dataset_directory)
        self.augmentation = augmentation
        self.samples = zip(
            [f"{dataset_directory / images_directory / data_file_name}{images_format}" for data_file_name in data_files_names_list],
            padding_labels(get_data_labels(data_files_names_list, df_labels), maximum_label_length),
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, sample_label = self.samples[idx]
        data = self.__load_image_func(sample_path)
        data = self.__dataset_specific_preprocesses(
            data,
            self.__chosen_dataset_specific_preprocesses_name_list,
            self.__chosen_dataset_specific_preprocesses_hyperparameters,
        )
        data = apply_preprocesses(
            image=data,
            chosen_common_preprocesses_list=self.__chosen_common_preprocesses_list,
            preprocesses_hyperparameters_kwargs=self.__chosen_common_preprocesses_hyperparameters_dict,
            seed=self._seed,
        )
        data = torch.from_numpy(data)
        if self.augmentation is not None:
            data = self.augmentation(data)
        return data.float(), sample_label

    def _log_chosen_preprocesses_info(
        self,
        chosen_preprocesses_name_list,
        chosen_preprocesses_hyperparameters_dict,
        preprocesses_name_list_in_order_of_apply,
        preprocesses_and_their_hyperparameters_dict,
        preprocesses_type,
    ):
        chosen_preprocesses_name_list.sort(key=preprocesses_name_list_in_order_of_apply.index)
        self._logger.info(f"Order of applying {preprocesses_type} preprocesses:")
        for i, preprocess_name in enumerate(preprocesses_name_list_in_order_of_apply):
            self._logger.info(f"{i + 1}- {preprocess_name}")
        self._logger.info(f"Chosen {preprocesses_type} preprocesses in order: {chosen_preprocesses_name_list}")
        if chosen_preprocesses_name_list:
            self._logger.info(f"{preprocesses_type.capitalize()} preprocesses hyperparameters:")
            for preprocess_name in chosen_preprocesses_name_list:
                if preprocess_name in preprocesses_and_their_hyperparameters_dict:
                    for hyperparameter_name in preprocesses_and_their_hyperparameters_dict[preprocess_name]:
                        self._logger.info(
                            f"{hyperparameter_name.replace('_', ' ').capitalize()}: {chosen_preprocesses_hyperparameters_dict[hyperparameter_name]}"
                        )
