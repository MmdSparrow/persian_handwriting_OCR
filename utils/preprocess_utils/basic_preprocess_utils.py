from constants.base_directories_names_constants import PROCESSED_DATA_DIRECTORY_NAME
from utils.load_and_store_dataset_labels_utils import get_data_labels, read_csv_file
from utils.load_and_store_folds_and_split_info import store_folds_info, store_train_validation_test_splits_info

from ._split_data_func import (
    split_test_and_others_func,
    split_test_train_validation_func,
    stratified_folds_split_func,
)

__all__ = [
    "split_stratified_folds",
    "split_test_and_train_and_validation",
]


def split_test_and_train_and_validation(
    dataset_directory,
    directory_to_store_splits_info_file,
    test_ratio,
    validation_ratio,
    seed,
):
    df_labels = read_csv_file(dataset_directory)
    data_files_paths_list = __list_npy_data_files_paths(dataset_directory=dataset_directory)
    labels_list = get_data_labels(data_files_paths_list, df_labels)
    data_files_paths_test, data_files_paths_train, data_files_paths_validation, _, _, _ = split_test_train_validation_func(
        data_files_paths_list,
        labels_list,
        test_ratio,
        validation_ratio,
        seed,
    )
    splits_info_file_path = store_train_validation_test_splits_info(
        directory_to_store_splits_info_file,
        data_files_paths_train,
        data_files_paths_validation,
        data_files_paths_test,
    )
    return splits_info_file_path


def split_stratified_folds(
    dataset_directory,
    directory_to_store_folds_and_splits_info_file,
    test_ratio,
    folds_num,
    seed,
):
    df_labels = read_csv_file(dataset_directory)
    data_files_paths_list = __list_npy_data_files_paths(dataset_directory=dataset_directory)
    labels_list = get_data_labels(data_files_paths_list, df_labels)
    data_files_paths_test, data_files_paths_train_and_validation, _, train_and_validation_labels = split_test_and_others_func(
        data_files_paths_list,
        labels_list,
        test_ratio,
        seed,
    )
    folds = stratified_folds_split_func(
        folds_num,
        train_and_validation_labels,
        seed,
    )
    folds_info_file_path, splits_info_file_path = store_folds_info(
        data_files_paths_train_and_validation,
        data_files_paths_test,
        folds,
        directory_to_store_folds_and_splits_info_file,
    )
    return folds_info_file_path, splits_info_file_path


def __list_npy_data_files_paths(dataset_directory):
    files_directories_list = []
    for processed_file in (dataset_directory / PROCESSED_DATA_DIRECTORY_NAME).iterdir():
        if processed_file.is_file() and processed_file.suffix == ".npy":
            files_directories_list.append(processed_file)
    return sorted(files_directories_list)
