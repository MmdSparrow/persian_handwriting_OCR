from ._dataset_specific_preprocess_parent import DatasetSpecificPreprocessParent

__all__ = ["TypedWordsImagesSpecificPreprocesses"]


class TypedWordsImagesSpecificPreprocesses(DatasetSpecificPreprocessParent):
    _CHOSEN_DATASET_SPECIFIC_PREPROCESSES_NAME_LIST_IN_ORDER_OF_APPLY = []
    _CHOSEN_DATASET_SPECIFIC_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT = {}

    def __init__(
        self,
        chosen_dataset_specific_preprocesses_name_list,
        chosen_dataset_specific_preprocesses_hyperparameters_dict,
        logger,
    ):
        super().__init__(chosen_dataset_specific_preprocesses_name_list, chosen_dataset_specific_preprocesses_hyperparameters_dict, logger)

    def apply_specific_prepreocesses(self, data):
        return data
