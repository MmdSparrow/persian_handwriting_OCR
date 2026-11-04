from sklearn.metrics import confusion_matrix

from constants.common_constants import (
    ACCURACY_SCORE_NAME,
    AUC_SCORE_NAME,
    BALANCED_ACCURACY_NAME,
    F1_SCORE_NAME,
    MACRO_F1_SCORE_NAME,
    MACRO_PRECISION_NAME,
    MACRO_RECALL_NAME,
    PRECISION_SCORE_NAME,
    PREDICTION_BASED_SCORES_NAMES_LIST,
    PROBABILITY_BASED_SCORES_NAMES_LIST,
    SENSITIVITY_SCORE_NAME,
    SPECIFICITY_SCORE_NAME,
    WEIGHTED_F1_SCORE_NAME,
)
from utils.experiment_utils.scores_functions import (
    accuracy_calculator,
    auc_score_calculator,
    balanced_accuracy_calculator,
    f1_score_calculator,
    macro_f1_score_calculator,
    macro_precision_calculator,
    macro_recall_calculator,
    precision_calculator,
    sensitivity_calculator,
    specificity_calculator,
    weighted_f1_score_calculator,
)

__all__ = [
    "compute_prediction_based_scores",
    "compute_probability_based_scores",
    "score_functions_selector",
    "store_channel_importance_counts",
]


def score_functions_selector(scores_name_list):
    SCORES_NAME_FUNCTION_MAP = {
        ACCURACY_SCORE_NAME: accuracy_calculator,
        SENSITIVITY_SCORE_NAME: sensitivity_calculator,
        SPECIFICITY_SCORE_NAME: specificity_calculator,
        F1_SCORE_NAME: f1_score_calculator,
        PRECISION_SCORE_NAME: precision_calculator,
        AUC_SCORE_NAME: auc_score_calculator,
        BALANCED_ACCURACY_NAME: balanced_accuracy_calculator,
        MACRO_F1_SCORE_NAME: macro_f1_score_calculator,
        WEIGHTED_F1_SCORE_NAME: weighted_f1_score_calculator,
        MACRO_RECALL_NAME: macro_recall_calculator,
        MACRO_PRECISION_NAME: macro_precision_calculator,
    }
    scores_functions_dict = {}
    for score_name in scores_name_list:
        try:
            scores_functions_dict[score_name] = SCORES_NAME_FUNCTION_MAP[score_name]
        except KeyError:
            raise ValueError(f"Unknown score: {score_name}. Available score: {list(SCORES_NAME_FUNCTION_MAP.keys())}")
    return scores_functions_dict


def compute_prediction_based_scores(predictions, labels, scores_functions_dict):
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    context = {
        "predictions": predictions,
        "labels": labels,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    scores = {}
    for score_name, score_function in scores_functions_dict.items():
        if score_name in PREDICTION_BASED_SCORES_NAMES_LIST:
            scores[score_name] = score_function(**context)
    return scores


def compute_probability_based_scores(probabilities, labels, scores_functions_dict):
    context = {
        "probabilities": probabilities,
        "labels": labels,
    }
    scores = {}
    for score_name, score_function in scores_functions_dict.items():
        if score_name in PROBABILITY_BASED_SCORES_NAMES_LIST:
            scores[score_name] = score_function(**context)
    return scores


def store_channel_importance_counts(
    first_20_percent_channels_count_true_prediction,
    second_20_percent_channels_count_true_prediction,
    first_20_percent_channels_count_false_prediction,
    second_20_percent_channels_count_false_prediction,
    total_samples_true_prediction,
    total_samples_flase_prediction,
    split_name,
    directory_to_save_files,
):
    with open(directory_to_save_files / split_name / "channels_importance_counts.txt", "w") as file_to_write:
        __write_channel_importance_counts_in_file(
            channels_count=first_20_percent_channels_count_true_prediction,
            is_first=True,
            is_prediction_true=True,
            total_samples=total_samples_true_prediction,
            file_to_write=file_to_write,
        )
        __write_channel_importance_counts_in_file(
            channels_count=second_20_percent_channels_count_true_prediction,
            is_first=False,
            is_prediction_true=True,
            total_samples=total_samples_true_prediction,
            file_to_write=file_to_write,
        )
        __write_channel_importance_counts_in_file(
            channels_count=first_20_percent_channels_count_false_prediction,
            is_first=True,
            is_prediction_true=False,
            total_samples=total_samples_flase_prediction,
            file_to_write=file_to_write,
        )
        __write_channel_importance_counts_in_file(
            channels_count=second_20_percent_channels_count_false_prediction,
            is_first=False,
            is_prediction_true=False,
            total_samples=total_samples_flase_prediction,
            file_to_write=file_to_write,
        )


def __write_channel_importance_counts_in_file(
    channels_count,
    is_first,
    is_prediction_true,
    total_samples,
    file_to_write,
):

    first_or_second_str = "First" if is_first else "second"
    file_to_write.write(f"{first_or_second_str} 20% Most Important Channels - {is_prediction_true} Prediction\n")
    file_to_write.write("=" * 50 + "\n")

    for channel, count in channels_count.most_common():
        percentage = 100 * count / total_samples
        file_to_write.write(f"{channel:<10}{count:<10}{percentage:>8.2f}%\n")
