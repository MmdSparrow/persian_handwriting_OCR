from constants.common_constants import (
    CHARACTER_ACCURACY,
    CHARACTER_ERROR_RATE,
    WORD_ACCURACY,
    WORD_ERROR_RATE,
)
from utils.experiment_utils.scores_functions import (
    character_accuracy_calculator,
    character_error_rate_calculator,
    word_accuracy_calculator,
    word_error_rate_calculator,
)

__all__ = [
    "compute_scores",
    "score_functions_selector",
]


def score_functions_selector(scores_name_list):
    SCORES_NAME_FUNCTION_MAP = {
        CHARACTER_ERROR_RATE: character_error_rate_calculator,
        CHARACTER_ACCURACY: character_accuracy_calculator,
        WORD_ERROR_RATE: word_error_rate_calculator,
        WORD_ACCURACY: word_accuracy_calculator,
    }
    scores_functions_dict = {}
    for score_name in scores_name_list:
        try:
            scores_functions_dict[score_name] = SCORES_NAME_FUNCTION_MAP[score_name]
        except KeyError:
            raise ValueError(f"Unknown score: {score_name}. Available score: {list(SCORES_NAME_FUNCTION_MAP.keys())}")
    return scores_functions_dict


def compute_scores(predictions, ground_truth, scores_functions_dict):
    scores = {}
    for score_name, score_function in scores_functions_dict.items():
        scores[score_name] = score_function(ground_truth, predictions)
    return scores
