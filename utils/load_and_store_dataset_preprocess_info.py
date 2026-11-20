import json

__all__ = [
    "load_dataset_preprocesses_info",
    "store_dataset_preprocesses_info",
]


__CHOSEN_COMMON_PREPROCESSES_LIST_KEY = "chosen_preprocesses_names_list"
__CHOSEN_COMMON_PREPROCESSES_HYPERPARAMETERS_DICT_KEY = "chosen_common_preprocesses_hyperparameters_dict"

__DATASET_PREPROCESSES_AND_FEATURE_EXTRACTION_FILE_NAME = "preprocess_and_feature_extraction_info.json"


def store_dataset_preprocesses_info(
    chosen_preprocesses_names_list,
    chosen_common_preprocesses_hyperparameters_dict,
    directories_to_store_info_file,
):
    data = {
        __CHOSEN_COMMON_PREPROCESSES_LIST_KEY: chosen_preprocesses_names_list,
        __CHOSEN_COMMON_PREPROCESSES_HYPERPARAMETERS_DICT_KEY: chosen_common_preprocesses_hyperparameters_dict,
    }
    for directory_to_store_info_file in directories_to_store_info_file:
        with open(directory_to_store_info_file / __DATASET_PREPROCESSES_AND_FEATURE_EXTRACTION_FILE_NAME, "w") as file_to_write:
            json.dump(data, file_to_write, indent=4)


def load_dataset_preprocesses_info(stored_info_file_directory):
    metadata_file_path = stored_info_file_directory / __DATASET_PREPROCESSES_AND_FEATURE_EXTRACTION_FILE_NAME
    with open(metadata_file_path, "r") as file_to_read:
        data = json.load(file_to_read)
    chosen_preprocesses_names_list = data[__CHOSEN_COMMON_PREPROCESSES_LIST_KEY]
    chosen_common_preprocesses_hyperparameters_dict = data[__CHOSEN_COMMON_PREPROCESSES_HYPERPARAMETERS_DICT_KEY]
    return chosen_preprocesses_names_list, chosen_common_preprocesses_hyperparameters_dict
