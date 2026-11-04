import numpy as np


def store_image(
    image,
    image_name,
    output_directory,
):
    data_file_path = output_directory / f"{image_name}.npy"
    np.save(data_file_path, image)
