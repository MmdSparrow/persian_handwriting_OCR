import cv2

__all__ = [
    "image_loader_selector",
]


def image_loader_selector(image_format):
    if image_format == ".tif" or image_format == ".tiff":
        return __load_tif_image


def __load_tif_image(image_path):
    return cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
