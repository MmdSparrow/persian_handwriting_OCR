import json

__all__ = [
    "get_train_and_validation_data_files_paths_for_fold_split",
    "get_train_and_validation_data_files_paths_for_test_and_train_and_validation_split",
    "store_folds_info",
    "store_train_validation_test_splits_info",
]

__SPLITS_INFO_FILE_NAME = "splits_info.json"
__FOLDS_INFO_FILE_NAME = "folds_info.json"

__SPLIT_INFO_TEST_KEY = "test"
__SPLIT_INFO_TRAIN_KEY = "train"
__SPLIT_INFO_VALIDATION_KEY = "validation"
__SPLIT_TRAIN_AND_VALIDATION_KEY = "trian_and_validation"


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
                __SPLIT_INFO_TEST_KEY: data_files_names_test,
                __SPLIT_INFO_TRAIN_KEY: data_files_names_train,
                __SPLIT_INFO_VALIDATION_KEY: data_files_names_validation,
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
                __SPLIT_INFO_TEST_KEY: data_files_names_test,
                __SPLIT_TRAIN_AND_VALIDATION_KEY: data_files_names_train_and_validation,
            },
            f,
            indent=4,
        )

    with open(folds_info_file_path, "w") as f:
        json.dump(folds, f, indent=4)

    return folds_info_file_path, splits_info_file_path


def get_train_and_validation_data_files_paths_for_fold_split(
    fold_index,
    folds_info_file_path,
    split_info_file_path,
):
    with open(folds_info_file_path, "r") as file_to_read:
        folds = json.load(file_to_read)
    with open(split_info_file_path, "r") as file_to_read:
        train_and_validation_files_paths = json.load(file_to_read)[__SPLIT_TRAIN_AND_VALIDATION_KEY]
    train_data_files_paths = [train_and_validation_files_paths[i] for i in folds[fold_index][__SPLIT_INFO_TRAIN_KEY]]
    validation_data_files_paths = [train_and_validation_files_paths[i] for i in folds[fold_index][__SPLIT_INFO_VALIDATION_KEY]]
    return train_data_files_paths, validation_data_files_paths


def get_train_and_validation_data_files_paths_for_test_and_train_and_validation_split(
    split_info_file_path,
):
    with open(split_info_file_path, "r") as file_to_read:
        train_and_validation_files_paths = json.load(file_to_read)
    return train_and_validation_files_paths[__SPLIT_INFO_TRAIN_KEY], train_and_validation_files_paths[__SPLIT_INFO_VALIDATION_KEY]
