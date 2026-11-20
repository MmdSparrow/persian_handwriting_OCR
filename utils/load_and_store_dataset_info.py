import json

__all__ = [
    "load_dataset_info",
    "store_dataset_info",
]


__IMAGES_FORMAT_KEY = "images_format"
__IMAGES_DIRECTORY_KEY = "images_directory"

__DATASET_INFO_FILE_NAME = "dataset_info.json"


def store_dataset_info(
    images_format,
    images_directory,
    directories_to_store_info_file,
):
    data = {
        __IMAGES_FORMAT_KEY: images_format,
        __IMAGES_DIRECTORY_KEY: images_directory,
    }
    for directory_to_store_info_file in directories_to_store_info_file:
        with open(directory_to_store_info_file / __DATASET_INFO_FILE_NAME, "w") as file_to_write:
            json.dump(data, file_to_write, indent=4)


def load_dataset_info(stored_info_file_directory):
    metadata_file_path = stored_info_file_directory / __DATASET_INFO_FILE_NAME
    with open(metadata_file_path, "r") as file_to_read:
        data = json.load(file_to_read)
    images_format = data[__IMAGES_FORMAT_KEY]
    images_directory = data[__IMAGES_DIRECTORY_KEY]
    return images_format, images_directory
