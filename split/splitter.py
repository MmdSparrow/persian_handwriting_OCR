from utils.load_and_store_dataset_info import load_dataset_info
from utils.preprocess_utils.split_utils import split_folds, split_test_and_train_and_validation

__all__ = ["Splitter"]


class Splitter:
    def __init__(self, logger, seed):
        self._logger = logger
        self._seed = seed

    def _log_preparation_details(self, test_ratio, validation_ratio):
        self._logger.info("Preparation hyperparameters:")
        self._logger.info(f"Test ratio={test_ratio}, Train ratio={1.0 - (test_ratio + validation_ratio)}, Validation ratio={validation_ratio}")

    def _log_preparation_details_fold(self, test_ratio, folds_num):
        self._logger.info("Preparation hyperparameters:")
        self._logger.info(f"Test ratio={test_ratio}, Folds number={folds_num}")

    def split_test_and_train_and_validation(
        self,
        dataset_directory,
        test_ratio,
        validation_ratio,
        directory_to_store_splits_info_file,
    ):
        self._log_preparation_details(test_ratio, validation_ratio)
        images_format, images_directory, _ = load_dataset_info(dataset_directory)
        return split_test_and_train_and_validation(
            dataset_directory=dataset_directory,
            images_directory=images_directory,
            images_format=images_format,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
            seed=self._seed,
            directory_to_store_splits_info_file=directory_to_store_splits_info_file,
        )

    def split_stratified_folds(
        self,
        dataset_directory,
        test_ratio,
        folds_num,
        directory_to_store_folds_and_splits_info_file,
    ):
        self._log_preparation_details_fold(test_ratio, folds_num)
        images_format, images_directory, _ = load_dataset_info(dataset_directory)
        return split_folds(
            dataset_directory=dataset_directory,
            images_directory=images_directory,
            images_format=images_format,
            test_ratio=test_ratio,
            folds_num=folds_num,
            seed=self._seed,
            directory_to_store_folds_and_splits_info_file=directory_to_store_folds_and_splits_info_file,
        )
