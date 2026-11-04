from utils.dataset_common_utils.modma_dataset_common_utils import load_and_prepare_data

from ._basic_preprocess_parent import BasicPreprocessParent

__all__ = ["BasicPreprocessTypedWordsImagesDataset"]


class BasicPreprocessTypedWordsImagesDataset(BasicPreprocessParent):
    def __init__(self, typed_word_image_dataset_directory, preprocessed_data_directory, logger, seed):
        super().__init__(typed_word_image_dataset_directory, preprocessed_data_directory, logger, seed)

    def _load_data(self, data_file_path):
        return load_and_prepare_data(data_file_path)
