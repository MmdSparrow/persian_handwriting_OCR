from ._base_parent import BaseParent

__all__ = ["BaseData"]


class BaseData(BaseParent):
    def __init__(self, dataset_directory, verbose=True, outputs_directory=None, seed=None):
        super().__init__(verbose, outputs_directory)
        self._dataset_directory = dataset_directory
        self._seed = seed

    def create_preprocessed_data_required_directories(self, pipeline):
        self.__create_preprocessed_data_directory(pipeline)
