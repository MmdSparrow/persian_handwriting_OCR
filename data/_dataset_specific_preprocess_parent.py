from abc import ABC, abstractmethod


class DatasetSpecificPreprocessParent(ABC):
    _CHOSEN_DATASET_SPECIFIC_PREPROCESSES_NAME_LIST_IN_ORDER_OF_APPLY = None
    _CHOSEN_DATASET_SPECIFIC_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT = None

    def __init__(self, logger):
        super().__init__()
        self._logger = logger

    @abstractmethod
    def apply_specific_prepreocesses(
        data,
        chosen_dataset_specific_preprocesses_name_list,
        chosen_dataset_specific_preprocesses_hyperparameters,
    ):
        pass

