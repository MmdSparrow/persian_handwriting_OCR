import json

from constants.common_constants import LOSS_KEY, SCORES_KEY

__all__ = [
    "load_loss",
    "load_scores",
    "store_loss",
    "store_scores",
]

__LOSS_FILE_NAME = "loss.json"
__SCORES_FILE_NAME = "scores.json"


def store_loss(split_name, loss, directory_to_save_results):
    with open(directory_to_save_results / split_name / __LOSS_FILE_NAME, "w") as f:
        json.dump({LOSS_KEY: loss}, f, indent=4)


def store_scores(split_name, scores, directory_to_save_results):
    with open(directory_to_save_results / split_name / __SCORES_FILE_NAME, "w") as f:
        json.dump({SCORES_KEY: scores}, f, indent=4)


def load_loss(results_directory, split_name):
    data_loss = None
    with open(results_directory / split_name / __LOSS_FILE_NAME, "r") as f:
        data_loss = json.load(f)
    return data_loss[LOSS_KEY]


def load_scores(results_directory, split_name):
    data_scores = None
    with open(results_directory / split_name / __SCORES_FILE_NAME, "r") as f:
        data_scores = json.load(f)
    return data_scores[SCORES_KEY]
