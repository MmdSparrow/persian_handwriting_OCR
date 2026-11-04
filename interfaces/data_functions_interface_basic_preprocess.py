from _base import BaseData
from utils.interfaces_utils import (
    basic_preprocess_class_selector,
    initial_preparation_class_selector,
    measure_runtime,
)

__all__ = ["DataFunctionsInterfaceBasicPreprocess"]


class DataFunctionsInterfaceBasicPreprocess(BaseData):
    def __init__(self, dataset_root_directory, verbose=True, output_path=None, seed=None, log_init_info=False, **kwargs):
        super().__init__(dataset_root_directory, verbose, output_path, seed, **kwargs)
        self.__dataset_name = self._dataset_directory.name
        if log_init_info:
            self._logger.info(f"Selected dataset is {self.__dataset_name}")
            if seed is not None:
                self._logger.info(f"Data functions seed is set to {seed}")
            else:
                self._logger.info("Data functions seed is not set")

    @measure_runtime("Initial preparation")
    def restruct_dataset(self):
        self.create_splits_directories()
        InitialPreparationClass = initial_preparation_class_selector(self.__dataset_name)
        initial_preparation = InitialPreparationClass(dataset_root_directory=self._dataset_directory)
        initial_preparation.restruct_dataset()
        self._logger.info("Dataset restructed")
        return self._splited_data_directory

    @measure_runtime("Preprocess")
    def preprocess(
        self,
        dataset_directory,
        chosen_preprocesses_list,
        preprocesses_hyperparameters_kwargs,
        delete_raw_data_files,
    ):
        self.create_splits_directories()
        BasicPreprocessClass = basic_preprocess_class_selector(self.__dataset_name)
        basic_preprocess = BasicPreprocessClass(
            dataset_directory=dataset_directory,
            preprocessed_data_directory=self._preprocessed_data_directory,
            logger=self._logger,
            seed=self._seed,
        )
        preprocessed_data_directory = basic_preprocess.apply_preprocesses(
            chosen_preprocesses_list=chosen_preprocesses_list,
            preprocesses_hyperparameters_kwargs=preprocesses_hyperparameters_kwargs,
            delete_raw_data_files=delete_raw_data_files,
        )
        self._logger.info("Preprocessing completed")
        return preprocessed_data_directory
