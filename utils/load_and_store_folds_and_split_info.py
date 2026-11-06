import json

from constants.common_constants import SPLIT_TEST_NAME, SPLIT_TRAIN_AND_VALIDATION_NAME, SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME

__all__ = [
    "store_folds_info",
    "store_train_validation_test_splits_info",
]

__SPLITS_INFO_FILE_NAME = "splits_info.json"
__FOLDS_INFO_FILE_NAME = "folds_info.json"


def store_train_validation_test_splits_info(
    outputs_directory,
    files_paths_train,
    files_paths_validation,
    files_paths_test,
):
    splits_info_file_path = outputs_directory / __SPLITS_INFO_FILE_NAME
    with open(splits_info_file_path, "w") as f:
        json.dump(
            {
                SPLIT_TEST_NAME: [str(x) for x in files_paths_test],
                SPLIT_TRAIN_NAME: [str(x) for x in files_paths_train],
                SPLIT_VALIDATION_NAME: [str(x) for x in files_paths_validation],
            },
            f,
            indent=4,
        )
    return splits_info_file_path


def store_folds_info(
    files_paths_train_and_validation,
    files_paths_test,
    folds,
    outputs_directory,
):
    splits_info_file_path = outputs_directory / __SPLITS_INFO_FILE_NAME
    folds_info_file_path = outputs_directory / __FOLDS_INFO_FILE_NAME

    with open(splits_info_file_path, "w") as f:
        json.dump(
            {
                SPLIT_TEST_NAME: [str(x) for x in files_paths_test],
                SPLIT_TRAIN_AND_VALIDATION_NAME: [str(x) for x in files_paths_train_and_validation],
            },
            f,
            indent=4,
        )

    with open(folds_info_file_path, "w") as f:
        json.dump(folds, f, indent=4)

    return folds_info_file_path, splits_info_file_path


def get_train_and_validation_data_files_paths(
    fold_index,
    folds_info_file_path,
    split_info_file_path,
):
    with open(folds_info_file_path, "r") as file_to_read:
        folds = json.load(file_to_read)
    with open(split_info_file_path, "r") as file_to_read:
        train_and_validation_files_paths = json.load(file_to_read)[SPLIT_TRAIN_AND_VALIDATION_NAME]
    return folds[fold_index]
