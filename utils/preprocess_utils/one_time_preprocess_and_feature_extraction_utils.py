import os
import shutil
from pathlib import Path

from constants.common_constants import (
    SPLIT_TEST_NAME,
    SPLIT_TRAIN_NAME,
    SPLIT_VALIDATION_NAME,
)

from ._common_one_time_preprocess_func import list_npy_data_files_paths
from ._common_preprocess_func import (
    split_test_and_others_func,
    split_test_train_validation_func,
    store_folds_info,
    store_train_validation_test_split_info,
    stratified_folds_split_func,
)

__all__ = [
    "split_each_data_to_a_file",
    "split_test_and_train_and_validation",
    "split_train_and_validation_for_input_fold",
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
    delete_preprocessed_data_files,
    seed,
):
    
    data_files_paths_list = list_npy_data_files_paths(dataset_root=dataset_root)
    labels_list=
    files_directories_test, files_directories_train, files_directories_validation, _, _, _ = split_test_train_validation_func(
        data_files_paths_list,
        labels_list,
        test_ratio,
        validation_ratio,
        seed,
    )

    split_info_file_path = store_train_validation_test_split_info(
        output_directory,
        files_directories_train,
        files_directories_validation,
        files_directories_test,
    )
    return split_info_file_path


def stratified_folds_split(
    splited_dataset_directory,
    directory_to_save_final_data,
    output_directory,
    test_ratio,
    folds_num,
    seed,
):
    shutil.copy(splited_dataset_directory / "dataset_metadata.json", directory_to_save_final_data)
    files_directories_list, labels_list = list_npy_data_files_paths(splited_dataset_directory)
    files_directories_test, files_directories_train_and_validation, _, train_and_validation_labels = split_test_and_others_func(
        files_directories_list,
        labels_list,
        test_ratio,
        seed,
    )
    test_dir = directory_to_save_final_data / SPLIT_TEST_NAME
    for files_directory in files_directories_test:
        for processed_file in files_directory.iterdir():
            shutil.copy2(processed_file, test_dir / f"{files_directory.name}_{processed_file.name}")

    folds = stratified_folds_split_func(
        folds_num,
        train_and_validation_labels,
        seed,
    )
    folds_info_file_path, split_info_file_path = store_folds_info(
        files_directories_train_and_validation,
        files_directories_test,
        folds,
        output_directory,
    )

    split_train_and_validation_for_input_fold(directory_to_save_final_data, folds[0], files_directories_train_and_validation)
    return folds_info_file_path, split_info_file_path


def split_train_and_validation_for_input_fold(directory_to_save_final_data, fold, files_directories_train_and_validation):
    train_dir = directory_to_save_final_data / SPLIT_TRAIN_NAME
    validation_dir = directory_to_save_final_data / SPLIT_VALIDATION_NAME

    files_directories_train = [Path(files_directories_train_and_validation[i]) for i in fold[SPLIT_TRAIN_NAME]]
    files_directories_validation = [Path(files_directories_train_and_validation[i]) for i in fold[SPLIT_VALIDATION_NAME]]

    for files_directory in files_directories_train:
        for processed_file in files_directory.iterdir():
            prefix = "_".join(files_directory.name.split("_")[:-1])
            shutil.copy2(processed_file, train_dir / f"{prefix}_{processed_file.name}")
    for files_directory in files_directories_validation:
        for processed_file in files_directory.iterdir():
            prefix = "_".join(files_directory.name.split("_")[:-1])
            shutil.copy2(processed_file, validation_dir / f"{prefix}_{processed_file.name}")
