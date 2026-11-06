from _base import BaseData
from utils.interfaces_utils import (
    basic_preprocess_class_selector,
    initial_preparation_class_selector,
    measure_runtime,
)

__all__ = ["DataFunctionsInterfaceBasicPreprocess"]


class DataFunctionsInterfaceBasicPreprocess(BaseData):
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
    def restruct_dataset(self):
        self.create_splits_directories()
        InitialPreparationClass = initial_preparation_class_selector(self.__dataset_name)
        initial_preparation = InitialPreparationClass(dataset_directory=self._dataset_directory)
        initial_preparation.restruct_dataset()
        self._logger.info("Dataset restructed")

    @measure_runtime("Preprocessing")
    def preprocess(
        self,
        chosen_preprocesses_list,
        preprocesses_hyperparameters_kwargs,
        delete_raw_data_files,
    ):
        self.create_splits_directories()
        BasicPreprocessClass = basic_preprocess_class_selector(self.__dataset_name)
        basic_preprocess = BasicPreprocessClass(
            logger=self._logger,
            seed=self._seed,
        )
        basic_preprocess.apply_preprocesses(
            dataset_directory=self._dataset_directory,
            directory_to_store_preprocessed_data=self._preprocessed_data_directory,
            chosen_preprocesses_list=chosen_preprocesses_list,
            preprocesses_hyperparameters_kwargs=preprocesses_hyperparameters_kwargs,
            delete_raw_data_files=delete_raw_data_files,
        )
        self._logger.info("Preprocessing completed")
        return self._preprocessed_data_directory

    @measure_runtime("Data splitting")
    def split_test_and_train_and_validation(
        self,
        test_ratio,
        validation_ratio,
    ):
        self.create_splits_directories()
        BasicPreprocessClass = basic_preprocess_class_selector(self.__dataset_name)
        basic_preprocess = BasicPreprocessClass(
            logger=self._logger,
            seed=self._seed,
        )
        splits_info_file_path = basic_preprocess.split_test_and_train_and_validation(
            preprocessed_data_directory=self._preprocessed_data_directory,
            directory_to_store_splits_info_file=self._outputs_directory,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
        )
        self._logger.info(f"Train-validation-test splitting completed successfully (test ratio: {test_ratio:.2f}, validation ratio: {validation_ratio:.2f})")
        return splits_info_file_path

    # def replace_split_info
    # TODO

    @measure_runtime("Data splitting")
    def split_stratified_folds(
        self,
        test_ratio,
        folds_num,
    ):
        self.create_splits_directories()
        BasicPreprocessClass = basic_preprocess_class_selector(self.__dataset_name)
        basic_preprocess = BasicPreprocessClass(
            logger=self._logger,
            seed=self._seed,
        )
        folds_info_file_path, splits_info_file_path = basic_preprocess.split_stratified_folds(
            preprocessed_data_directory=self._preprocessed_data_directory,
            directory_to_store_folds_and_splits_info_file=self._outputs_directory,
            test_ratio=test_ratio,
            folds_num=folds_num,
        )
        self._logger.info(f"Stratified splitting completed successfully: {folds_num} folds generated with a test ratio of {test_ratio:.2f}")
        return folds_info_file_path, splits_info_file_path

    # def replace_fold_info
    # TODO
