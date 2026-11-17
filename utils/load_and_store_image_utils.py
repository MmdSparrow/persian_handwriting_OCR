import numpy as np


def store_image(
    image,
    image_name,
    directory_to_store_image,
):
    data_file_path = directory_to_store_image / f"{image_name}.npy"
    np.save(data_file_path, image)
