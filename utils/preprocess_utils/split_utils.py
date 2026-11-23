import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split

from constants.common_constants import SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME
from utils.load_and_store_folds_and_split_info import store_folds_info, store_train_validation_test_splits_info

__all__ = [
    "split_folds",
    "split_test_and_train_and_validation",
]

__EQUAL_TO_ZERO_THRESHOLD = 1e-7


def split_test_and_train_and_validation(
    dataset_directory,
    images_directory,
    images_format,
    test_ratio,
    validation_ratio,
    seed,
    directory_to_store_splits_info_file,
):
    data_files_names_list = __list_data_files_names(dataset_directory, images_directory, images_format)
    data_files_names_test, train_and_validation_data_list = __split_by_ratio_func(
        data_files_names_list,
        test_ratio,
        seed,
    )
    data_files_names_train, data_files_names_validation = __split_by_ratio_func(
        train_and_validation_data_list,
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


def split_folds(
    dataset_directory,
    images_directory,
    images_format,
    test_ratio,
    folds_num,
    seed,
    directory_to_store_folds_and_splits_info_file,
):
    data_files_names_list = __list_data_files_names(dataset_directory, images_directory, images_format)
    data_files_names_test, data_files_names_train_and_validation = __split_by_ratio_func(
        data_files_names_list,
        test_ratio,
        seed,
    )
    folds = __folds_split_func(
        folds_num,
        len(data_files_names_train_and_validation),
        seed,
    )
    folds_info_file_path, splits_info_file_path = store_folds_info(
        data_files_names_train_and_validation,
        data_files_names_test,
        folds,
        directory_to_store_folds_and_splits_info_file,
    )
    return folds_info_file_path, splits_info_file_path


def __folds_split_func(
    folds_num,
    data_list_length,
    seed,
):
    if folds_num == "all":
        splitter = LeaveOneOut()
    else:
        splitter = KFold(
            n_splits=folds_num,
            shuffle=True,
            random_state=seed,
        )
    folds = list(splitter.split(np.zeros(data_list_length)))
    folds = [
        {
            SPLIT_TRAIN_NAME: train_idx.tolist(),
            SPLIT_VALIDATION_NAME: val_idx.tolist(),
        }
        for train_idx, val_idx in folds
    ]
    return folds


def __split_by_ratio_func(
    data_list,
    ratio,
    seed,
):
    first_data_list, second_data_list = None, None
    if ratio is None or ratio < __EQUAL_TO_ZERO_THRESHOLD:
        first_data_list = []
        second_data_list = data_list
    else:
        first_data_list, second_data_list = train_test_split(
            data_list,
            test_size=ratio,
            random_state=seed,
            shuffle=True,
        )
    return first_data_list, second_data_list


def __list_data_files_names(
    dataset_directory,
    images_directory,
    images_format,
):
    return [data_file_path.stem for data_file_path in (dataset_directory / images_directory).glob(f"*{images_format}")]
    # data_files_names_list, _ = read_csv_file(dataset_directory)
    # return data_files_names_list
