import os
import tarfile
from pathlib import Path

import requests

from constants.base_directories_names_constants import DATASETS_ROOT_DIRECTORY_NAME
from constants.common_constants import TYPED_WORDS_IMAGES_DATASET_NAME

__all__ = ["donwload_dataset"]

__DATASET_PATH_ON_COLAB = Path("/content") / "OCR_persian_text" / DATASETS_ROOT_DIRECTORY_NAME
__DATASET_PATH_ON_LOCAL = DATASETS_ROOT_DIRECTORY_NAME

__TYPED_WORDS_IMAGES_DATASET_COUNT = 120000


def donwload_dataset(dataset_name, colab_version=True):
    dataset_root_directory = None
    if colab_version:
        dataset_root_directory = __DATASET_PATH_ON_COLAB
    else:
        dataset_root_directory = __DATASET_PATH_ON_LOCAL
    os.makedirs(dataset_root_directory, exist_ok=True)

    if dataset_name == TYPED_WORDS_IMAGES_DATASET_NAME:
        __download_typed_words_images(dataset_root_directory)
    else:
        raise ValueError()


def __download_typed_words_images(dataset_root_directory):
    __URL = "https://github.com/MmdSparrow/typed_words_images_dataset/raw/refs/heads/master/typed_words_images.tar.xz"
    destination_dir = dataset_root_directory / TYPED_WORDS_IMAGES_DATASET_NAME
    destination_dir.mkdir(parents=True, exist_ok=True)
    archive_path = destination_dir / "typed_words_images.tar.xz"
    response = requests.get(__URL, stream=True)
    response.raise_for_status()
    with open(archive_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
    with tarfile.open(archive_path, mode="r:xz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts

            if len(parts) > 1:
                member.name = str(Path(*parts[1:]))
                tar.extract(member, path=destination_dir)
    archive_path.unlink()
