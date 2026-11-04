import numpy as np

__all__ = [
    "compute_prediction_by_voting_for_chunks",
    "generate_report_machine_learning",
]


def generate_report_machine_learning(scores_dict, split):
    base_str = split.capitalize() + ": "
    for key in scores_dict:
        base_str += f"{key}: {scores_dict[key]:.7f}, "
    base_str = base_str[:-2]
    return base_str


def compute_prediction_by_voting_for_chunks(output_logit_chunks_list):
    predictions = []
    for chunk_predictions in output_logit_chunks_list:
        chunk_predictions = np.asarray(chunk_predictions)
        classes, counts = np.unique(chunk_predictions, return_counts=True)
        prediction = classes[np.argmax(counts)]
        predictions.append(prediction)
    return predictions
