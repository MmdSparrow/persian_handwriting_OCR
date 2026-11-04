from pathlib import Path

__all__ = [
    "check_mne_fsaverage_data_is_presented",
]


def check_mne_fsaverage_data_is_presented():
    root_path = Path.home()
    mne_fsaverage_data_directory = root_path / "mne_data" / "MNE-fsaverage-data" / "fsaverage"
    required_folders = ["bem", "label", "mri", "mri2m", "scripts", "surf"]
    all_exist = all((mne_fsaverage_data_directory / folder).is_dir() for folder in required_folders)
    return all_exist
