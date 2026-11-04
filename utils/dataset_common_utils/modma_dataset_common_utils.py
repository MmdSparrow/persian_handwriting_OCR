import shutil

import numpy as np
import pandas as pd
from scipy.io import loadmat

from info.modma_dataset_info import MODMA_DATASET_REST_DATA_DIRECTORY_NAME

__all__ = [
    "delete_other_files",
    "list_raw_data_and_labels_func",
]


def delete_other_files(modma_raw_dataset_root):
    for item in modma_raw_dataset_root.iterdir():
        if item.name == MODMA_DATASET_REST_DATA_DIRECTORY_NAME or item.name == "splited":
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def list_raw_data_and_labels_func(modma_raw_dataset_root):
    selected_dataset_root = modma_raw_dataset_root / MODMA_DATASET_REST_DATA_DIRECTORY_NAME / MODMA_DATASET_REST_DATA_DIRECTORY_NAME
    files_paths_list = []
    labels_list = []
    metadata_path = next(f for f in selected_dataset_root.iterdir() if str(f).endswith(".xlsx"))
    metadata_df = pd.read_excel(metadata_path, dtype={"subject id": str})

    for signal_file in selected_dataset_root.iterdir():
        if signal_file.suffix == ".mat":
            subject_id = signal_file.stem[:8]
            subject_metadata = metadata_df.loc[metadata_df["subject id"] == subject_id]
            label = 1 if subject_metadata["type"].iloc[0] == "MDD" else 0
            files_paths_list.append(signal_file)
            labels_list.append(label)

    sorted_pairs = sorted(zip(files_paths_list, labels_list))
    files_paths_list, labels_list = map(list, zip(*sorted_pairs))
    return files_paths_list, labels_list, metadata_df


def load_and_prepare_data(file_path, metadata_df=None):
    data = loadmat(file_path)
    att_name = next((key for key in data if key.endswith("mat")), None)
    data = np.asarray(data[att_name], dtype=np.float32)[:128, :]
    if metadata_df is None:
        return data, None
    subject_id = file_path.stem[:8]
    subject_metadata = metadata_df.loc[metadata_df["subject id"] == subject_id]
    label = 1 if subject_metadata["type"].iloc[0] == "MDD" else 0
    return data, label
