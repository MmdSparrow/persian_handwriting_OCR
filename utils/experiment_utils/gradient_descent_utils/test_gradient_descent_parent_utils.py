from constants.common_constants import SPLIT_TEST_NAME
from utils.load_and_store_loss_and_scores import store_loss, store_scores

__all__ = [
    "generate_final_report",
    "save_final_results",
]


def save_final_results(avg_total_loss, avg_scores, directory_to_save_results):
    store_loss(
        SPLIT_TEST_NAME,
        avg_total_loss,
        directory_to_save_results,
    )
    store_scores(
        SPLIT_TEST_NAME,
        avg_scores,
        directory_to_save_results,
    )


def generate_final_report(scores_dict, avg_total_loss):
    base_str = f"Final Result - Test: Avg Loss: {avg_total_loss:.7f}"
    for key in scores_dict:
        base_str += f", {key}: {scores_dict[key]:.7f}"
    return base_str
