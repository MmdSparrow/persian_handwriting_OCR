import json

__all__ = [
    "load_dataset_metadata",
    "store_dataset_metadata",
]


__CHOSEN_PREPROCESSES_LIST_KEY = "chosen_preprocesses_list"
__PREPROCESSES_HYPERPARAMETERS_KWARGS_KEY = "preprocesses_hyperparameters_kwargs"

__DATASET_META_DATA_FILE_NAME = "dataset_metadata.json"


def store_dataset_metadata(
    chosen_preprocesses_list,
    directory_to_save,
    preprocesses_hyperparameters_kwargs,
):
    data = {
        __CHOSEN_PREPROCESSES_LIST_KEY: chosen_preprocesses_list,
        __PREPROCESSES_HYPERPARAMETERS_KWARGS_KEY: preprocesses_hyperparameters_kwargs,
    }
    with open(directory_to_save / __DATASET_META_DATA_FILE_NAME, "w") as file_to_write:
        json.dump(data, file_to_write, indent=4)


def load_dataset_metadata(directory_to_load):
    metadata_file_path = directory_to_load / __DATASET_META_DATA_FILE_NAME
    with open(metadata_file_path, "r") as file_to_read:
        data = json.load(file_to_read)
    chosen_preprocesses_list = data[__CHOSEN_PREPROCESSES_LIST_KEY]
    preprocesses_hyperparameters_kwargs = data[__PREPROCESSES_HYPERPARAMETERS_KWARGS_KEY]
    return chosen_preprocesses_list, preprocesses_hyperparameters_kwargs
