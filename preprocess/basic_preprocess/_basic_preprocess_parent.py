import shutil
from abc import ABC, abstractmethod

from constants.base_directories_names_constants import DATASET_IMAGES_DIRECTORY_NAME
from preprocess._preprocess_parent import PreprocessParent
from utils.load_and_store_data_utils import store_image
from utils.load_and_store_dataset_metadata import store_dataset_metadata
from utils.preprocess_utils.basic_preprocess_utils import split_stratified_folds, split_test_and_train_and_validation
from utils.preprocess_utils.preprocess_function import apply_preprocesses

__all__ = ["BasicPreprocessParent"]


class BasicPreprocessParent(ABC, PreprocessParent):
    def __init__(self, logger, seed):
        super().__init__(logger, seed)

    @abstractmethod
    def _load_data(self, data_file_path):
        pass

    def apply_preprocesses(
        self,
        dataset_directory,
        directory_to_store_preprocessed_data,
        chosen_preprocesses_list,
        preprocesses_hyperparameters_kwargs,
        delete_raw_data_files,
    ):
        self._log_dataset_info(self._CHANNELS_RESPONSIVE_TO_BLINKING, self._EOG_REMOVAL_ICA_N_COMPONENTS)
        self._log_preprocesses_info(chosen_preprocesses_list, **preprocesses_hyperparameters_kwargs)
        for data_file_path in list((dataset_directory / DATASET_IMAGES_DIRECTORY_NAME).iterdir()):
            image = self._load_data(data_file_path)
            image = apply_preprocesses(
                image=image,
                chosen_common_preprocesses_list=chosen_preprocesses_list,
                preprocesses_hyperparameters_kwargs=preprocesses_hyperparameters_kwargs,
                seed=self._seed,
            )
            store_image(
                image=image,
                image_name=data_file_path.stem,
                directory_to_store_image=directory_to_store_preprocessed_data / DATASET_IMAGES_DIRECTORY_NAME,
            )
            if delete_raw_data_files:
                shutil.rmtree(data_file_path)

        if delete_raw_data_files:
            shutil.rmtree(dataset_directory / DATASET_IMAGES_DIRECTORY_NAME)

        store_dataset_metadata(
            chosen_preprocesses_list=chosen_preprocesses_list,
            directory_to_save=directory_to_store_preprocessed_data,
            preprocesses_hyperparameters_kwargs=preprocesses_hyperparameters_kwargs,
        )

    def split_test_and_train_and_validation(
        self,
        preprocessed_data_directory,
        directory_to_store_splits_info_file,
        test_ratio,
        validation_ratio,
    ):
        self._log_preparation_details(test_ratio, validation_ratio)
        return split_test_and_train_and_validation(
            dataset_directory=preprocessed_data_directory,
            directory_to_store_splits_info_file=directory_to_store_splits_info_file,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
            seed=self._seed,
        )

    def split_stratified_folds(
        self,
        preprocessed_data_directory,
        directory_to_store_folds_and_splits_info_file,
        test_ratio,
        folds_num,
    ):
        self._log_preparation_details_fold(test_ratio, folds_num)
        return split_stratified_folds(
            dataset_directory=preprocessed_data_directory,
            directory_to_store_folds_and_splits_info_file=directory_to_store_folds_and_splits_info_file,
            test_ratio=test_ratio,
            folds_num=folds_num,
            seed=self._seed,
        )
