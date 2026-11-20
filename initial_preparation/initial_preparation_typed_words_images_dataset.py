from info.typed_words_images import IMAGES_DIRECTORY, IMAGES_FORMAT

from ._initial_preparation_parent import InitialPreparationParent


class InitialPreparationTypedWrdsImgesDatasets(InitialPreparationParent):
    __IMAGES_FORMAT = IMAGES_FORMAT
    __IMAGES_DIRECTORY = IMAGES_DIRECTORY

    def __init__(self, dataset_directory, logger):
        super().__init__(dataset_directory, logger)

    def _list_data_names_and_labels(self):
        pass

    def _generate_data_labels(self):
        pass
