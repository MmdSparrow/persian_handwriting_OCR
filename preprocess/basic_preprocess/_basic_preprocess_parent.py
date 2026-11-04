import json
import shutil
from abc import ABC, abstractmethod

from constants.base_directories_names_constants import DATASET_IMAGES_DIRECTORY_NAME, PROCESSED_DATA_DIRECTORY_NAME, RESULTS_DIRECTORY_NAME, SPLITED_DATA_DIRECTORY_NAME
from constants.common_constants import SPLIT_TRAIN_AND_VALIDATION_NAME
from preprocess._preprocess_parent import PreprocessParent
from utils.load_and_store_data_utils import store_image
from utils.load_and_store_dataset_metadata import store_dataset_metadata
from utils.preprocess_utils.one_time_preprocess_and_feature_extraction_utils import (
    split_test_and_train_and_validation,
    split_train_and_validation_for_input_fold,
    stratified_folds_split,
)
from utils.preprocess_utils.preprocess_function import apply_preprocesses

__all__ = ["BasicPreprocessParent"]


class BasicPreprocessParent(ABC, PreprocessParent):
    def __init__(self, dataset_directory, preprocessed_data_directory, logger, seed):
        super().__init__(logger, seed)
        self._dataset_directory = dataset_directory
        self._preprocessed_data_directory = preprocessed_data_directory

    @abstractmethod
    def _load_data(self, data_file_path):
        pass

    def apply_preprocesses(
        self,
        chosen_preprocesses_list,
        delete_raw_data_files,
        preprocesses_hyperparameters_kwargs,
    ):
        self._log_dataset_info(self._CHANNELS_RESPONSIVE_TO_BLINKING, self._EOG_REMOVAL_ICA_N_COMPONENTS)
        self._log_preprocesses_info(chosen_preprocesses_list, **preprocesses_hyperparameters_kwargs)
        for data_file_path in list((self._dataset_directory / DATASET_IMAGES_DIRECTORY_NAME).iterdir()):
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
                output_directory=self._preprocessed_data_directory / DATASET_IMAGES_DIRECTORY_NAME,
            )
            if delete_raw_data_files:
                shutil.rmtree(data_file_path)

        if delete_raw_data_files:
            shutil.rmtree(self._dataset_directory / DATASET_IMAGES_DIRECTORY_NAME)

        store_dataset_metadata(
            chosen_preprocesses_list=chosen_preprocesses_list,
            directory_to_save=self._preprocessed_data_directory,
            preprocesses_hyperparameters_kwargs=preprocesses_hyperparameters_kwargs,
        )

    def split_test_and_train_and_validation(
        self,
        dataset_root,
        output_directory,
        test_ratio,
        validation_ratio,
        delete_preprocessed_data_files,
    ):
        self._log_preparation_details(test_ratio, validation_ratio)
        return split_test_and_train_and_validation(
            dataset_root=dataset_root,
            output_directory=output_directory,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
            delete_preprocessed_data_files=delete_preprocessed_data_files,
            seed=self._seed,
        )

    def split_stratified_folds_version(
        self,
        preprocessed_data_directory,
        directory_to_save_splits,
        output_directory,
        test_ratio,
        folds_num,
    ):
        self._log_preparation_details_fold_version(test_ratio, folds_num)
        return stratified_folds_split(
        )

    def prepare_next_fold(
        self,
        directory_to_save_final_data,
        folds_info_file_path,
        current_fold,
        file_paths_train_and_validation_path,
    ):
        fold_index = current_fold + 1
        folds = None
        files_directories_train_and_validation = None
        with open(folds_info_file_path, "r") as f:
            folds = json.load(f)
        with open(file_paths_train_and_validation_path, "r") as f:
            files_directories_train_and_validation = json.load(f)[SPLIT_TRAIN_AND_VALIDATION_NAME]
        split_train_and_validation_for_input_fold(
            directory_to_save_final_data=directory_to_save_final_data,
            fold=folds[fold_index],
            files_directories_train_and_validation=files_directories_train_and_validation,
        )
