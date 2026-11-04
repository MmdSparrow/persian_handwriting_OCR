import json


def count_folds_func(folds_info_file_path):
    with open(folds_info_file_path, "r") as file_to_read:
        folds = json.load(file_to_read)
    return len(folds)
