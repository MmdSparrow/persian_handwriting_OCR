__all__ = ["apply_preprocesses"]


import cv2
import numpy as np

from constants.common_constants import (
    DENOISING_METHODS_MEDIAN_BLUR,
    PREPROCESS_CONVERT_TO_GRAY_SCALE,
    PREPROCESS_CROP_TEXT_REGION,
    PREPROCESS_DENOISING,
    PREPROCESS_DESKEW,
    PREPROCESS_HYPERPARAMETER_DENOISING_METHOD,
    PREPROCESS_HYPERPARAMETER_RESIZING_TARGET_HEIGHT,
    PREPROCESS_HYPERPARAMETER_RESIZING_TARGET_WIDTH,
    PREPROCESS_NORMALIZE_PIXEL_VALUES,
    PREPROCESS_RESIZE,
)


def apply_preprocesses(
    image,
    chosen_common_preprocesses_list,
    preprocesses_hyperparameters_kwargs,
    seed,
):
    if PREPROCESS_CONVERT_TO_GRAY_SCALE in chosen_common_preprocesses_list and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if PREPROCESS_DENOISING in chosen_common_preprocesses_list:
        image = __denoising_method_selector(image, preprocesses_hyperparameters_kwargs[PREPROCESS_HYPERPARAMETER_DENOISING_METHOD])
    if PREPROCESS_CROP_TEXT_REGION in chosen_common_preprocesses_list:
        pass
    if PREPROCESS_RESIZE in chosen_common_preprocesses_list:
        if len(image.shape) == 3:
            _, original_height, original_width = image.shape
        else:
            original_height, original_width = image.shape
        scale = preprocesses_hyperparameters_kwargs[PREPROCESS_HYPERPARAMETER_RESIZING_TARGET_HEIGHT] / original_height
        new_width = int(original_width * scale)
        new_width = min(new_width, preprocesses_hyperparameters_kwargs[PREPROCESS_HYPERPARAMETER_RESIZING_TARGET_WIDTH])
        image = cv2.resize(
            image, (new_width, preprocesses_hyperparameters_kwargs[PREPROCESS_HYPERPARAMETER_RESIZING_TARGET_HEIGHT]), interpolation=cv2.INTER_AREA
        )
    if PREPROCESS_DESKEW in chosen_common_preprocesses_list:
        pass
    if PREPROCESS_NORMALIZE_PIXEL_VALUES in chosen_common_preprocesses_list:
        image = image.astype(np.float32) / 255.0
    return image
    # # Add channel dimension
    # tensor = tensor.unsqueeze(0)


def __denoising_method_selector(image, denoising_method):
    if denoising_method == DENOISING_METHODS_MEDIAN_BLUR:
        return cv2.medianBlur(image, 3)
