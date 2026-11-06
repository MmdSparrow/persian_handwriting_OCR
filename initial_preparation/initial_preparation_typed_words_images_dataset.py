from constants.common_constants import TYPED_WORDS_IMAGES_DATASET_NAME

from ._initial_preparation_parent import InitialPreparationParent


class InitialPreparationTypedWrdsImgesDatasets(InitialPreparationParent):
    _DATASET_NAME = TYPED_WORDS_IMAGES_DATASET_NAME

    def __init__(self, dataset_directory):
        super().__init__(dataset_directory)

    def _list_data_paths_and_labels(self):
        pass

    def restruct_dataset(self):
        pass
