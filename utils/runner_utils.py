import json

from constants.common_constants import SPLIT_TRAIN_AND_VALIDATION_NAME, SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME

__all__ = [
    "count_folds_func",
    "get_train_and_validation_data_files_paths_for_fold_split",
    "get_train_and_validation_data_files_paths_for_test_and_train_and_validation_split",
]


def count_folds_func(folds_info_file_path):
    with open(folds_info_file_path, "r") as file_to_read:
        folds = json.load(file_to_read)
    return len(folds)


def get_train_and_validation_data_files_paths_for_fold_split(
    fold_index,
    folds_info_file_path,
    split_info_file_path,
):
    with open(folds_info_file_path, "r") as file_to_read:
        folds = json.load(file_to_read)
    with open(split_info_file_path, "r") as file_to_read:
        train_and_validation_files_paths = json.load(file_to_read)[SPLIT_TRAIN_AND_VALIDATION_NAME]
    train_data_files_paths = [train_and_validation_files_paths[i] for i in folds[fold_index][SPLIT_TRAIN_NAME]]
    validation_data_files_paths = [train_and_validation_files_paths[i] for i in folds[fold_index][SPLIT_VALIDATION_NAME]]
    return train_data_files_paths, validation_data_files_paths


def get_train_and_validation_data_files_paths_for_test_and_train_and_validation_split(
    split_info_file_path,
):
    with open(split_info_file_path, "r") as file_to_read:
        train_and_validation_files_paths = json.load(file_to_read)[SPLIT_TRAIN_AND_VALIDATION_NAME]
    return train_and_validation_files_paths[SPLIT_TRAIN_NAME], train_and_validation_files_paths[SPLIT_VALIDATION_NAME]
