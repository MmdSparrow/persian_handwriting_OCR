from ._dataset_specific_preprocess_parent import DatasetSpecificPreprocessParent

__all__ = ["TypedWordsImagesSpecificPreprocesses"]


class TypedWordsImagesSpecificPreprocesses(DatasetSpecificPreprocessParent):
    _CHOSEN_DATASET_SPECIFIC_PREPROCESSES_NAME_LIST_IN_ORDER_OF_APPLY = None
    _CHOSEN_DATASET_SPECIFIC_PREPROCESSES_AND_ITS_HYPERPARAMETERS_DICT = None

    def __init__(self, logger):
        super().__init__(logger)
