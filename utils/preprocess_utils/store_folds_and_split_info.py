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
    data_files_names_train,
    data_files_names_validation,
    data_files_names_test,
):
    splits_info_file_path = outputs_directory / __SPLITS_INFO_FILE_NAME
    with open(splits_info_file_path, "w") as f:
        json.dump(
            {
                SPLIT_TEST_NAME: data_files_names_test,
                SPLIT_TRAIN_NAME: data_files_names_train,
                SPLIT_VALIDATION_NAME: data_files_names_validation,
            },
            f,
            indent=4,
        )
    return splits_info_file_path


def store_folds_info(
    data_files_names_train_and_validation,
    data_files_names_test,
    folds,
    outputs_directory,
):
    splits_info_file_path = outputs_directory / __SPLITS_INFO_FILE_NAME
    folds_info_file_path = outputs_directory / __FOLDS_INFO_FILE_NAME

    with open(splits_info_file_path, "w") as f:
        json.dump(
            {
                SPLIT_TEST_NAME: data_files_names_test,
                SPLIT_TRAIN_AND_VALIDATION_NAME: data_files_names_train_and_validation,
            },
            f,
            indent=4,
        )

    with open(folds_info_file_path, "w") as f:
        json.dump(folds, f, indent=4)

    return folds_info_file_path, splits_info_file_path
