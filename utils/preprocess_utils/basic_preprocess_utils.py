import os
import shutil

from constants.base_directories_names_constants import PROCESSED_DATA_DIRECTORY_NAME
from utils.load_and_store_dataset_labels_utils import get_data_labels, read_csv_file

from ._split_data_func import (
    split_test_and_others_func,
    split_test_train_validation_func,
    store_folds_info,
    store_train_validation_test_split_info,
    stratified_folds_split_func,
)

__all__ = [
    "split_each_data_to_a_file",
    "split_test_and_train_and_validation",
    "stratified_folds_split",
]


def split_each_data_to_a_file(dataset_root_directory, output_root_directory, file_format, copy=False):
    transfer_function = shutil.copy if copy else shutil.move
    for signal_file in dataset_root_directory.iterdir():
        if signal_file.suffix == file_format:
            output_directory = output_root_directory / signal_file.stem
            os.makedirs(output_directory, exist_ok=False)
            transfer_function(signal_file, output_directory)


def split_test_and_train_and_validation(
    dataset_root,
    output_directory,
    test_ratio,
    validation_ratio,
    seed,
):
    df_labels = read_csv_file(dataset_root)
    data_files_paths_list = __list_npy_data_files_paths(dataset_root=dataset_root)
    labels_list = get_data_labels(data_files_paths_list, df_labels)
    data_files_paths_test, data_files_paths_train, data_files_paths_validation, _, _, _ = split_test_train_validation_func(
        data_files_paths_list,
        labels_list,
        test_ratio,
        validation_ratio,
        seed,
    )
    split_info_file_path = store_train_validation_test_split_info(
        output_directory,
        data_files_paths_train,
        data_files_paths_validation,
        data_files_paths_test,
    )
    return split_info_file_path


def stratified_folds_split(
    dataset_root,
    output_directory,
    test_ratio,
    folds_num,
    seed,
):
    df_labels = read_csv_file(dataset_root)
    data_files_paths_list = __list_npy_data_files_paths(dataset_root=dataset_root)
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
    folds_info_file_path, split_info_file_path = store_folds_info(
        data_files_paths_train_and_validation,
        data_files_paths_test,
        folds,
        output_directory,
    )
    return folds_info_file_path, split_info_file_path


def __list_npy_data_files_paths(dataset_root):
    files_directories_list = []
    for processed_file in (dataset_root / PROCESSED_DATA_DIRECTORY_NAME).iterdir():
        if processed_file.is_file() and processed_file.suffix == ".npy":
            files_directories_list.append(processed_file)
    return sorted(files_directories_list)
