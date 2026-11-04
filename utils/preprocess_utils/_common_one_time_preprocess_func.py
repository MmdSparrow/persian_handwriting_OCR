__all__ = ["list_npy_data_files_paths"]

from constants.base_directories_names_constants import PROCESSED_DATA_DIRECTORY_NAME


def list_npy_data_files_paths(dataset_root):
    files_directories_list = []
    for processed_file in (dataset_root / PROCESSED_DATA_DIRECTORY_NAME).iterdir():
        if processed_file.is_file() and processed_file.suffix == ".npy":
            files_directories_list.append(processed_file)
    return sorted(files_directories_list)
