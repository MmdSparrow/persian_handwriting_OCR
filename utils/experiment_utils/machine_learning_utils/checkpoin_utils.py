import json

import joblib

__all__ = [
    "load_checkpoint",
    "save_checkpoint",
]

__MODEL_FILE_NAME = "model.joblib"
__CHECKPOINT_DETAILS_FILE_NAME = "checkpoint_details.json"


def load_checkpoint(checkpoint_directory):
    model_path = checkpoint_directory / __MODEL_FILE_NAME
    model = joblib.load(model_path)
    return model


__all__ = []


def save_checkpoint(
    model,
    directory_to_save_checkpoint,
):
    joblib.dump(model, directory_to_save_checkpoint / __MODEL_FILE_NAME)
    with open(directory_to_save_checkpoint / __CHECKPOINT_DETAILS_FILE_NAME, "w") as f:
        json.dump(model.get_params(), f, indent=4)
