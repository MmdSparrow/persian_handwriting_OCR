import torch
from torch import nn

from constants.common_constants import CROSS_ENTROPY_LOSS, FOCAL_LOSS
from utils.experiment_utils.loss_functions import FocalLoss

__all__ = [
    "compute_avg_column_wise",
    "compute_chunks_batch_avg_loss",
    "compute_prediction",
    "compute_prediction_by_voting_on_chunks_batch",
    "compute_probability",
    "count_parameters",
    "generate_report_per_batch",
    "loss_function_selector",
]


def loss_function_selector(loss_function_name):
    LOSSES_NAME_FUNCTIONS_MAP = {
        CROSS_ENTROPY_LOSS: nn.CrossEntropyLoss(label_smoothing=0.1),
        FOCAL_LOSS: FocalLoss(gamma=2.0),
    }
    try:
        return LOSSES_NAME_FUNCTIONS_MAP[loss_function_name]
    except KeyError:
        raise ValueError(f"Unknown loss function: {loss_function_name}. Available losses: {list(LOSSES_NAME_FUNCTIONS_MAP.keys())}")


def generate_report_per_batch(batch_scores_dict, batch_loss, batch_index, batches_num, split):
    base_str = split.capitalize() + f"-Batch: [{batch_index + 1}/{batches_num}], Loss: {batch_loss:.7f}"
    for key in batch_scores_dict:
        base_str += f", {key}: {batch_scores_dict[key]:.7f}"
    return base_str


def compute_prediction(outputs_logit):
    return outputs_logit.argmax(dim=1)


def compute_prediction_by_voting_on_chunks_batch(output_logit_chunks_batch_list):
    chunks_predictions = torch.stack([torch.argmax(chunk, dim=1) for chunk in output_logit_chunks_batch_list]).T
    predictions, _ = torch.mode(chunks_predictions, dim=1)
    return predictions.tolist()


def compute_probability(outputs_logit):
    return outputs_logit.softmax(dim=1)[:, 1]


def compute_avg_column_wise(matrix_2d):
    return matrix_2d.mean(dim=0)


def compute_chunks_batch_avg_loss(output_logit_chunks_batch_list, label_batch, criterion):
    total_loss = 0.0
    for ouptut_chunk_batch in output_logit_chunks_batch_list:
        total_loss += criterion(ouptut_chunk_batch, label_batch).item()
    return total_loss / len(output_logit_chunks_batch_list)


def count_parameters(model, dataloader_train):
    device = next(model.parameters()).device
    with torch.no_grad():
        sample_batch = next(iter(dataloader_train))
        dummy_input = sample_batch[0][:1].to(device)
        _ = model(dummy_input)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
