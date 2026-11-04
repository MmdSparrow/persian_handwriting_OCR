import mne

from info.mumtaz_dataset_info import CHANNELS_NAME_MAPPING_DICT

__all__ = [
    "delete_other_files",
    "list_raw_data_and_labels_func",
]


def delete_other_files(mumtaz_raw_dataset_root, just_keep="EC"):
    ITEMS_TO_REMOVE_NAMES = [
        # DUPLICATE FILES
        "H S30 EC.edf",
        "MDD S34 EC.edf",
    ]
    for item in mumtaz_raw_dataset_root.iterdir():
        if not item.is_dir():
            if just_keep not in item.name:
                item.unlink()
            if item.name in ITEMS_TO_REMOVE_NAMES:
                item.unlink()


def list_raw_data_and_labels_func(mumtaz_raw_dataset_root):
    files_paths_list = []
    labels_list = []
    for signal_file in mumtaz_raw_dataset_root.iterdir():
        if signal_file.suffix == ".edf":
            label = 1 if "MDD" in signal_file.name else 0
            files_paths_list.append(signal_file)
            labels_list.append(label)

    sorted_pairs = sorted(zip(files_paths_list, labels_list))
    files_paths_list, labels_list = map(list, zip(*sorted_pairs))
    return files_paths_list, labels_list


def load_and_prepare_data(file_path):
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    channels_to_drop = [
        "EEG A2-A1",
        "EEG 23A-23R",
        "EEG 24A-24R",
    ]
    raw.drop_channels([ch for ch in channels_to_drop if ch in raw.ch_names])
    raw.rename_channels(CHANNELS_NAME_MAPPING_DICT)
    data = raw.get_data()
    label = 1 if "MDD" in file_path.name else 0
    return data, label
