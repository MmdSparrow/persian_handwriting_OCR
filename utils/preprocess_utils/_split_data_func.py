import numpy as np
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, train_test_split

from constants.common_constants import SPLIT_TRAIN_NAME, SPLIT_VALIDATION_NAME

__all__ = [
    "split_test_and_others_func",
    "split_test_train_validation_func",
    "stratified_folds_split_func",
]

__EQUAL_TO_ZERO_THRESHOLD = 1e-7


def stratified_folds_split_func(
    folds_num,
    labels,
    seed,
):
    if folds_num == "all":
        splitter = LeaveOneOut()
    else:
        splitter = StratifiedKFold(
            n_splits=folds_num,
            shuffle=True,
            random_state=seed,
        )
    folds = list(splitter.split(np.zeros(len(labels)), labels))
    folds = [
        {
            SPLIT_TRAIN_NAME: train_idx.tolist(),
            SPLIT_VALIDATION_NAME: val_idx.tolist(),
        }
        for train_idx, val_idx in folds
    ]
    return folds


def split_test_and_others_func(
    files_paths_list,
    labels_list,
    test_ratio,
    seed,
):
    files_paths_train_and_validation, files_paths_test, train_and_validation_labels, test_labels = None, None, None, None
    if test_ratio is None or test_ratio < __EQUAL_TO_ZERO_THRESHOLD:
        files_paths_train_and_validation = files_paths_list
        files_paths_test = []
        train_and_validation_labels = labels_list
        test_labels = []
    else:
        files_paths_train_and_validation, files_paths_test, train_and_validation_labels, test_labels = train_test_split(
            files_paths_list,
            labels_list,
            test_size=test_ratio,
            stratify=labels_list,
            random_state=seed,
        )
    return files_paths_test, files_paths_train_and_validation, test_labels, train_and_validation_labels


def split_test_train_validation_func(
    files_paths_list,
    labels_list,
    test_ratio,
    validation_ratio,
    seed,
):
    files_paths_test, files_paths_train_and_validation, test_labels, train_and_validation_labels = split_test_and_others_func(
        files_paths_list,
        labels_list,
        test_ratio,
        seed,
    )
    files_paths_train, files_paths_validation, train_labels, validation_labels = None, None, None, None
    if validation_ratio is None or validation_ratio < __EQUAL_TO_ZERO_THRESHOLD:
        files_paths_train = files_paths_train_and_validation
        files_paths_validation = []
        train_labels = train_and_validation_labels
        validation_labels = []
    else:
        files_paths_train, files_paths_validation, train_labels, validation_labels = train_test_split(
            files_paths_train_and_validation,
            train_and_validation_labels,
            test_size=validation_ratio,
            stratify=train_and_validation_labels,
            random_state=seed,
        )
    return files_paths_test, files_paths_train, files_paths_validation, test_labels, train_labels, validation_labels
