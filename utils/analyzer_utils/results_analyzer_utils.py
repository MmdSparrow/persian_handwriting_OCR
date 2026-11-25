import os

from utils.common_utils import mean_std_calculator

__all__ = [
    "create_folds_directories_in_input_directory",
    "write_analyzed_statistics_of_folds_in_file",
]


def create_folds_directories_in_input_directory(folds_number, input_directory):
    folds_paths = [input_directory / "folds" /f"fold_{i}" for i in range(1, folds_number + 1)]
    for fold_path in folds_paths:
        os.makedirs(fold_path, exist_ok=True)
    return folds_paths


def write_analyzed_statistics_of_folds_in_file(
    split_name,
    folds_losses_list,
    folds_scores_list_of_dict,
    analyze_report_file_path,
):
    __DECIMAL_PLACES = 4

    folds_scores_dict = {}
    for key in folds_scores_list_of_dict[0]:
        folds_scores_dict[key] = [score[key] for score in folds_scores_list_of_dict]

    with open(analyze_report_file_path, "a") as f:
        f.write(split_name.capitalize() + " Statistics:\n")
        if folds_losses_list is not None:
            folds_losses_mean, folds_losses_std = mean_std_calculator(folds_losses_list)
            f.write(f"Loss= {folds_losses_mean:.{__DECIMAL_PLACES}f} \u00b1 {folds_losses_std:.{__DECIMAL_PLACES}f}\n")
        for key, value in folds_scores_dict.items():
            value_mean, value_std = mean_std_calculator(value)
            f.write(f"{key}= {value_mean * 100:.{__DECIMAL_PLACES}f}% \u00b1 {value_std * 100:.{__DECIMAL_PLACES}f}%\n")
        f.write("\n")
