from torch import optim

from constants.common_constants import (
    OPTIMIZER_ADAM_NAME,
    OPTIMIZER_ADAMW_NAME,
    OPTIMIZER_SGD_NAME,
    SCHEDULER_CONSTANT_NAME,
    SCHEDULER_PLATEAU_NAME,
    SPLIT_TRAIN_NAME,
    SPLIT_VALIDATION_NAME,
)
from utils.load_and_store_loss_and_scores import store_loss, store_scores

__all__ = [
    "FinalCheckpointDetails",
    "generate_final_checkpoint_report",
    "generate_report_per_epoch",
    "optimizer_selector",
    "save_final_results",
    "scheduler_selector",
]


class FinalCheckpointDetails:
    def __init__(self):
        self.epoch = None
        self.loss_train = None
        self.loss_validation = None
        self.scores_train = None
        self.scores_validation = None

    def set_params(self, epoch, loss_train, loss_validation, scores_train, scores_validation):
        self.epoch = epoch
        self.loss_train = loss_train
        self.loss_validation = loss_validation
        self.scores_train = scores_train
        self.scores_validation = scores_validation


def optimizer_selector(optimizer_name, model, learning_rate, optimizer_hyperparameters_dict=None):
    if optimizer_name == OPTIMIZER_SGD_NAME:
        if optimizer_hyperparameters_dict is None:
            return optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )
        else:
            return optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=optimizer_hyperparameters_dict["momentum"],
                weight_decay=optimizer_hyperparameters_dict["weight_decay"],
            )
    elif optimizer_name == OPTIMIZER_ADAM_NAME:
        if optimizer_hyperparameters_dict is None:
            return optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4,
            )
        else:
            return optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=optimizer_hyperparameters_dict["weight_decay"],
            )
    elif optimizer_name == OPTIMIZER_ADAMW_NAME:
        if optimizer_hyperparameters_dict is None:
            return optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4,
            )
        else:
            return optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=optimizer_hyperparameters_dict["weight_decay"],
            )
    else:
        raise ValueError("optimizer_name")


def scheduler_selector(scheduler_name, optimizer, scheduler_hyperparameters_dict):
    __INFINITY = 1e6
    if scheduler_name == SCHEDULER_CONSTANT_NAME:
        return optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=__INFINITY)
    elif scheduler_name == SCHEDULER_PLATEAU_NAME:
        if scheduler_hyperparameters_dict is None:
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", threshold=0.01, factor=0.7, patience=2)
        else:
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                threshold=scheduler_hyperparameters_dict["threshold"],
                factor=scheduler_hyperparameters_dict["factor"],
                patience=scheduler_hyperparameters_dict["patience"],
            )
    else:
        raise ValueError("scheduler_name")


def generate_report_per_epoch(epoch_scores_dict, epoch_loss, learning_rate, epoch_index, epochs_num, split):
    base_str = split.capitalize() + f"-Epoch: [{epoch_index + 1}/{epochs_num}], Avg Loss: {epoch_loss:.7f}, Learning Rate: {learning_rate:.9f}"
    for key in epoch_scores_dict:
        base_str += f", {key}: {epoch_scores_dict[key]:.7f}"
    return base_str


def generate_final_checkpoint_report(final_checkpoint_details: FinalCheckpointDetails):
    base_str = f"Final Chckpoint - Avg Loss Train: {final_checkpoint_details.loss_train:.7f}, Avg Loss Validation (= Min Loss): {final_checkpoint_details.loss_validation:.7f}, epoch: {final_checkpoint_details.epoch + 1}."
    base_str += " Train: "
    for key in final_checkpoint_details.scores_train:
        base_str += f"{key}: {final_checkpoint_details.scores_train[key]:.7f}, "
    base_str = base_str[:-2]
    base_str += ". Validation: "
    for key in final_checkpoint_details.scores_validation:
        base_str += f"{key}: {final_checkpoint_details.scores_validation[key]:.7f}, "
    base_str = base_str[:-2]
    return base_str


def save_final_results(loss_list_train, loss_list_validation, scores_list_train, scores_list_validation, directory_to_save_results):
    store_loss(
        SPLIT_TRAIN_NAME,
        loss_list_train,
        directory_to_save_results,
    )
    store_scores(
        SPLIT_TRAIN_NAME,
        scores_list_train,
        directory_to_save_results,
    )

    store_loss(
        SPLIT_VALIDATION_NAME,
        loss_list_validation,
        directory_to_save_results,
    )
    store_scores(
        SPLIT_VALIDATION_NAME,
        scores_list_validation,
        directory_to_save_results,
    )
