import shutil
from pathlib import Path

import pandas as pd

from constants.common_constants import SPLIT_TEST_NAME, SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME
from info.modma_dataset_info import MODMA_DATASET_REST_DATA_DIRECTORY_NAME
from utils.dataset_common_utils.modma_dataset_common_utils import list_raw_data_and_labels_func
from utils.preprocess_utils._common_preprocess_func import (
    split_test_and_others_func,
    split_test_train_validation_func,
    store_folds_info,
    store_train_validation_test_split_info,
    stratified_folds_split_func,
)

__all__ = [
    "split_stratified_folds",
    "split_test_and_train_and_validation",
    "split_train_and_validation_for_input_fold",
]


def split_test_and_train_and_validation(
    modma_raw_dataset_root,
    directory_to_save_splited_data,
    output_directory,
    test_ratio=0.2,
    validation_ratio=0.15,
    copy_files=False,
    seed=None,
):
    files_paths_list, labels_list, metadata_df = list_raw_data_and_labels_func(modma_raw_dataset_root)
    files_paths_test, files_paths_train, files_paths_validation, _, _, _ = split_test_train_validation_func(
        files_paths_list,
        labels_list,
        test_ratio,
        validation_ratio,
        seed,
    )

    transfer_function = shutil.copy2 if copy_files else shutil.move
    train_dir = directory_to_save_splited_data / SPLIT_TRAIN_NAME
    validation_dir = directory_to_save_splited_data / SPLIT_VALIDATION_NAME
    test_dir = directory_to_save_splited_data / SPLIT_TEST_NAME

    for file_path in files_paths_train:
        transfer_function(file_path, train_dir)
    for file_path in files_paths_validation:
        transfer_function(file_path, validation_dir)
    for file_path in files_paths_test:
        transfer_function(file_path, test_dir)

    train_metadata = metadata_df[metadata_df["subject id"].isin(path.stem[:8] for path in files_paths_train)]
    validation_metadata = metadata_df[metadata_df["subject id"].isin(path.stem[:8] for path in files_paths_validation)]
    test_metadata = metadata_df[metadata_df["subject id"].isin(path.stem[:8] for path in files_paths_test)]

    train_metadata.to_excel(train_dir / "metadata.xlsx", index=False)
    validation_metadata.to_excel(validation_dir / "metadata.xlsx", index=False)
    test_metadata.to_excel(test_dir / "metadata.xlsx", index=False)

    split_info_file_path = store_train_validation_test_split_info(
        output_directory,
        files_paths_train,
        files_paths_validation,
        files_paths_test,
    )
    return split_info_file_path


def split_stratified_folds(modma_raw_dataset_root, directory_to_save_splited_data, output_directory, test_ratio=0.2, folds_num=10, seed=None):
    files_paths_list, labels_list, metadata_df = list_raw_data_and_labels_func(modma_raw_dataset_root)
    files_paths_test, files_paths_train_and_validation, _, train_and_validation_labels = split_test_and_others_func(
        files_paths_list, labels_list, test_ratio, seed
    )

    test_dir = directory_to_save_splited_data / SPLIT_TEST_NAME
    for file_path in files_paths_test:
        shutil.copy2(file_path, test_dir / Path(file_path).name)
    test_metadata = metadata_df[metadata_df["subject id"].isin(path.stem[:8] for path in files_paths_test)]
    test_metadata.to_excel(test_dir / "metadata.xlsx", index=False)

    folds = stratified_folds_split_func(
        folds_num,
        train_and_validation_labels,
        seed,
    )
    folds_info_file_path, split_info_file_path = store_folds_info(
        files_paths_train_and_validation,
        files_paths_test,
        folds,
        output_directory,
    )

    split_train_and_validation_for_input_fold(modma_raw_dataset_root, directory_to_save_splited_data, folds[0], files_paths_train_and_validation)
    return folds_info_file_path, split_info_file_path


def split_train_and_validation_for_input_fold(modma_raw_dataset_root, directory_to_save_splited_data, fold, file_paths_train_and_validation):
    selected_dataset_root = modma_raw_dataset_root / MODMA_DATASET_REST_DATA_DIRECTORY_NAME / MODMA_DATASET_REST_DATA_DIRECTORY_NAME
    metadata_path = next(f for f in selected_dataset_root.iterdir() if str(f).endswith(".xlsx"))
    metadata_df = pd.read_excel(metadata_path, dtype={"subject id": str})

    train_dir = directory_to_save_splited_data / SPLIT_TRAIN_NAME
    validation_dir = directory_to_save_splited_data / SPLIT_VALIDATION_NAME

    file_paths_train = [file_paths_train_and_validation[i] for i in fold[SPLIT_TRAIN_NAME]]
    file_paths_validation = [file_paths_train_and_validation[i] for i in fold[SPLIT_VALIDATION_NAME]]

    for file_path in file_paths_train:
        shutil.copy2(file_path, train_dir / Path(file_path).name)
    for file_path in file_paths_validation:
        shutil.copy2(file_path, validation_dir / Path(file_path).name)

    train_metadata = metadata_df[metadata_df["subject id"].isin(Path(path).stem[:8] for path in file_paths_train)]
    validation_metadata = metadata_df[metadata_df["subject id"].isin(Path(path).stem[:8] for path in file_paths_validation)]

    train_metadata.to_excel(train_dir / "metadata.xlsx", index=False)
    validation_metadata.to_excel(validation_dir / "metadata.xlsx", index=False)
