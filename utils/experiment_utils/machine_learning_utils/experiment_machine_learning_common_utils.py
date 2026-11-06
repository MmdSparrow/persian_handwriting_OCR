import numpy as np

__all__ = [
    "compute_prediction_by_voting_for_chunks",
]



def compute_prediction_by_voting_for_chunks(output_logit_chunks_list):
    predictions = []
    for chunk_predictions in output_logit_chunks_list:
        chunk_predictions = np.asarray(chunk_predictions)
        classes, counts = np.unique(chunk_predictions, return_counts=True)
        prediction = classes[np.argmax(counts)]
        predictions.append(prediction)
    return predictions
