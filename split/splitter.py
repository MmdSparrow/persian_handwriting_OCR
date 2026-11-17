from utils.preprocess_utils.split_utils import split_stratified_folds, split_test_and_train_and_validation

from ._splitter_parent import SplitterParent

__all__ = ["Splitter"]


class Splitter(SplitterParent):
    def __init__(self, logger, seed):
        super().__init__(logger, seed)

    def split_test_and_train_and_validation(
        self,
        dataset_directory,
        images_directory,
        images_format,
        test_ratio,
        validation_ratio,
        directory_to_store_splits_info_file,
    ):
        self._log_preparation_details(test_ratio, validation_ratio)
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
        images_directory,
        images_format,
        test_ratio,
        folds_num,
        directory_to_store_folds_and_splits_info_file,
    ):
        self._log_preparation_details_fold(test_ratio, folds_num)
        return split_stratified_folds(
            dataset_directory=dataset_directory,
            images_directory=images_directory,
            images_format=images_format,
            test_ratio=test_ratio,
            folds_num=folds_num,
            seed=self._seed,
            directory_to_store_folds_and_splits_info_file=directory_to_store_folds_and_splits_info_file,
        )
