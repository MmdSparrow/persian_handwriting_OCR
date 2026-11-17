from utils.load_and_store_dataset_labels_utils import get_data_labels, read_csv_file

from ._split_func import (
    split_test_and_others_func,
    split_test_train_validation_func,
    stratified_folds_split_func,
)
from .store_folds_and_split_info import store_folds_info, store_train_validation_test_splits_info

__all__ = [
    "split_stratified_folds",
    "split_test_and_train_and_validation",
]


def split_test_and_train_and_validation(
    dataset_directory,
    images_directory,
    images_format,
    test_ratio,
    validation_ratio,
    seed,
    directory_to_store_splits_info_file,
):
    data_files_names_list, labels_list = __list_data_files_names_and_its_labels(dataset_directory, images_directory, images_format)
    data_files_names_test, data_files_names_train, data_files_names_validation, _, _, _ = split_test_train_validation_func(
        data_files_names_list,
        labels_list,
        test_ratio,
        validation_ratio,
        seed,
    )
    splits_info_file_path = store_train_validation_test_splits_info(
        directory_to_store_splits_info_file,
        data_files_names_train,
        data_files_names_validation,
        data_files_names_test,
    )
    return splits_info_file_path


def split_stratified_folds(
    dataset_directory,
    images_directory,
    images_format,
    test_ratio,
    folds_num,
    seed,
    directory_to_store_folds_and_splits_info_file,
):
    data_files_names_list, labels_list = __list_data_files_names_and_its_labels(dataset_directory, images_directory, images_format)
    data_files_names_test, data_files_names_train_and_validation, _, train_and_validation_labels = split_test_and_others_func(
        data_files_names_list,
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
        data_files_names_train_and_validation,
        data_files_names_test,
        folds,
        directory_to_store_folds_and_splits_info_file,
    )
    return folds_info_file_path, splits_info_file_path


def __list_data_files_names_and_its_labels(
    dataset_directory,
    images_directory,
    images_format,
):
    data_files_names_list = list((dataset_directory / images_directory).glob(f"*{images_format}"))
    labels_df = read_csv_file(dataset_directory)
    labels_list = get_data_labels(data_files_names_list, labels_df)
    return data_files_names_list, labels_list
