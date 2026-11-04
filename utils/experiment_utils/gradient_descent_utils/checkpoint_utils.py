import json

import torch

from constants.common_constants import (
    BATCH_SIZE_KEY,
    EPOCH_KEY,
    EPOCHS_NUM_KEY,
    LOSS_NAME_KEY,
    MIN_LOSS_KEY,
    MODEL_STATE_DICT_KEY,
    OPTIMIZER_NAME_KEY,
    OPTIMIZER_STATE_DICT_KEY,
    SCHEDULER_NAME_KEY,
    SCHEDULER_STATE_DICT_KEY,
    SPLIT_TRAIN_NAME,
    SPLIT_VALIDATION_NAME,
)
from utils.load_and_store_loss_and_scores import load_loss, load_scores

__all__ = [
    "load_checkpoint",
    "load_final_checkpoint",
    "save_checkpoint",
]

__MODEL_FILE_NAME = "model.pth"
__CHECKPOINT_DETAILS_FILE_NAME = "checkpoint_details.json"


def save_checkpoint(
    batch_size,
    epoch,
    epochs_num,
    model_state_dict,
    optimizer_name,
    optimizer_state_dict,
    scheduler_name,
    scheduler_state_dict,
    loss_function_name,
    final_checkpoint_details,
    min_loss,
    directory_to_save_checkpoint,
):
    torch.save(
        {
            BATCH_SIZE_KEY: batch_size,
            EPOCH_KEY: epoch,
            EPOCHS_NUM_KEY: epochs_num,
            MODEL_STATE_DICT_KEY: model_state_dict,
            OPTIMIZER_NAME_KEY: optimizer_name,
            OPTIMIZER_STATE_DICT_KEY: optimizer_state_dict,
            SCHEDULER_NAME_KEY: scheduler_name,
            SCHEDULER_STATE_DICT_KEY: scheduler_state_dict,
            LOSS_NAME_KEY: loss_function_name,
            MIN_LOSS_KEY: min_loss,
        },
        directory_to_save_checkpoint / __MODEL_FILE_NAME,
    )
    with open(directory_to_save_checkpoint / __CHECKPOINT_DETAILS_FILE_NAME, "w") as f:
        json.dump(final_checkpoint_details.__dict__, f, indent=4)


def load_final_checkpoint(checkpoint_directory):
    checkpoint = torch.load(checkpoint_directory / __MODEL_FILE_NAME, weights_only=False)
    return (
        checkpoint[MODEL_STATE_DICT_KEY],
        checkpoint[LOSS_NAME_KEY],
    )


def load_checkpoint(checkpoint_directory, results_directory):
    checkpoint = torch.load(checkpoint_directory / __MODEL_FILE_NAME, weights_only=False)

    train_data_loss = load_loss(results_directory, SPLIT_TRAIN_NAME)
    validation_data_loss = load_loss(results_directory, SPLIT_VALIDATION_NAME)
    train_data_scores = load_scores(results_directory, SPLIT_TRAIN_NAME)
    validation_data_scores = load_scores(results_directory, SPLIT_VALIDATION_NAME)

    checkpoint_details = None
    with open(checkpoint_directory / __CHECKPOINT_DETAILS_FILE_NAME, "r") as f:
        checkpoint_details = json.load(f)

    return (
        checkpoint[BATCH_SIZE_KEY],
        checkpoint[EPOCH_KEY],
        checkpoint[EPOCHS_NUM_KEY],
        checkpoint[MODEL_STATE_DICT_KEY],
        checkpoint[OPTIMIZER_NAME_KEY],
        checkpoint[OPTIMIZER_STATE_DICT_KEY],
        checkpoint[SCHEDULER_NAME_KEY],
        checkpoint[SCHEDULER_STATE_DICT_KEY],
        checkpoint[LOSS_NAME_KEY],
        checkpoint_details,
        checkpoint[MIN_LOSS_KEY],
        train_data_loss,
        validation_data_loss,
        train_data_scores,
        validation_data_scores,
    )
