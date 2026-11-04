import pandas as pd

from constants.base_directories_names_constants import DATA_LABELS_CSV_FILE_NAME

__all__ = [
    "generate_data_labels_csv_file",
    "get_data_label",
    "get_data_labels",
    "read_csv_file",
]

__DATA_COLUMN_NAME = "Data Name"
__LABEL_COLUMN_NAME = "Label"


def generate_data_labels_csv_file(data_names_list, labels_list, directory_to_save_csv_file):
    df = pd.DataFrame({__DATA_COLUMN_NAME: data_names_list, __LABEL_COLUMN_NAME: labels_list})
    df.to_csv(directory_to_save_csv_file / DATA_LABELS_CSV_FILE_NAME, index=False, encoding="utf-8")


def read_csv_file(csv_file_directory):
    return pd.read_csv(csv_file_directory / DATA_LABELS_CSV_FILE_NAME, encoding="utf-8"), __DATA_COLUMN_NAME, __LABEL_COLUMN_NAME


def get_data_label(data_path, labels_df):
    return labels_df.loc[labels_df[__DATA_COLUMN_NAME] == data_path.name, __LABEL_COLUMN_NAME].iloc[0]


def get_data_labels(data_paths_list, labels_df):
    return [labels_df.loc[labels_df[__DATA_COLUMN_NAME] == sample_path.name, __LABEL_COLUMN_NAME].iloc[0] for sample_path in data_paths_list]
