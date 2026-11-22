from info.typed_words_images_info import IMAGES_DIRECTORY, IMAGES_FORMAT, MAXIMUM_LABEL_LENGTH

from ._initial_preparation_parent import InitialPreparationParent


class InitialPreparationTypedWrdsImgesDatasets(InitialPreparationParent):
    _IMAGES_FORMAT = IMAGES_FORMAT
    _IMAGES_DIRECTORY = IMAGES_DIRECTORY
    _MAXIMUM_LABEL_LENGTH = MAXIMUM_LABEL_LENGTH

    def __init__(self, dataset_directory, logger):
        super().__init__(dataset_directory, logger)

    def _list_data_names_and_labels(self):
        pass

    def _generate_data_labels(self):
        pass
