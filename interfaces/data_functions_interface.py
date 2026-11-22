from _base import BaseData
from split import Splitter
from utils.interfaces_utils import initial_preparation_class_selector, measure_runtime

__all__ = ["DataFunctionsInterface"]


class DataFunctionsInterface(BaseData):
    def __init__(
        self,
        dataset_directory,
        verbose=True,
        output_path=None,
        seed=None,
        log_init_info=False,
    ):
        super().__init__(dataset_directory, verbose, output_path, seed)
        self.__dataset_name = self._dataset_directory.name
        if log_init_info:
            self._logger.info(f"Dataset {self.__dataset_name} is selected")
            if seed is not None:
                self._logger.info(f"Data functions seed is set to {seed}")
            else:
                self._logger.info("Data functions seed is not set")

    @measure_runtime("Initial preparation")
    def initial_preparation(self):
        InitialPreparationClass = initial_preparation_class_selector(self.__dataset_name)
        initial_preparation = InitialPreparationClass(dataset_directory=self._dataset_directory)
        initial_preparation.initial_preparation()
        self._logger.info("Dataset restructed")

    @measure_runtime("Data splitting")
    def split_test_and_train_and_validation(
        self,
        test_ratio,
        validation_ratio,
    ):
        splitter = Splitter(
            logger=self._logger,
            seed=self._seed,
        )
        splits_info_file_path = splitter.split_test_and_train_and_validation(
            dataset_directory=self._dataset_directory,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
            directory_to_store_splits_info_file=self._outputs_directory,
        )
        self._logger.info(f"Train-validation-test splitting completed successfully (test ratio: {test_ratio:.2f}, validation ratio: {validation_ratio:.2f})")
        return splits_info_file_path

    @measure_runtime("Data splitting")
    def split_stratified_folds(
        self,
        test_ratio,
        folds_num,
    ):
        splitter = Splitter(
            logger=self._logger,
            seed=self._seed,
        )
        folds_info_file_path, splits_info_file_path = splitter.split_stratified_folds(
            dataset_directory=self._dataset_directory,
            test_ratio=test_ratio,
            folds_num=folds_num,
            directory_to_store_folds_and_splits_info_file=self._outputs_directory,
        )
        self._logger.info(f"Stratified splitting completed successfully: {folds_num} folds generated with a test ratio of {test_ratio:.2f}")
        return folds_info_file_path, splits_info_file_path
