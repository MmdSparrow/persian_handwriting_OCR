import pandas as pd

__all__ = [
    "generate_data_labels_csv_file",
    "read_csv_file",
]

__DATA_LABELS_CSV_FILE_NAME = "labels.csv"

__DATA_COLUMN_NAME = "Data Name"
__LABEL_COLUMN_NAME = "Label"


def generate_data_labels_csv_file(data_names_list, labels_list, directory_to_save_csv_file):
    df = pd.DataFrame({__DATA_COLUMN_NAME: data_names_list, __LABEL_COLUMN_NAME: labels_list})
    df.to_csv(directory_to_save_csv_file / __DATA_LABELS_CSV_FILE_NAME, index=False, encoding="utf-8")


def read_csv_file(csv_file_directory):
    df = pd.read_csv(csv_file_directory / __DATA_LABELS_CSV_FILE_NAME, encoding="utf-8", dtype={__DATA_COLUMN_NAME: str})
    data_files_names_list = df[__DATA_COLUMN_NAME].tolist()
    labels_list = df[__LABEL_COLUMN_NAME].tolist()
    return data_files_names_list, labels_list

