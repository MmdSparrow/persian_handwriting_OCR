# from jiwer import cer, wer

__all__ = [
    "character_accuracy_calculator",
    "character_error_rate_calculator",
    "word_accuracy_calculator",
    "word_error_rate_calculator",
]


# def character_error_rate_calculator(ground_truth, predictions):
#     return cer(ground_truth, predictions)


# def word_error_rate_calculator(ground_truth, predictions):
#     return wer(ground_truth, predictions)


def character_error_rate_calculator(ground_truth, predictions):
    total_errors = 0
    total_characters = 0
    for gt, pred in zip(ground_truth, predictions):
        total_errors += __levenshtein_distance(gt, pred)
        total_characters += len(gt)
    return total_errors / total_characters if total_characters else 0.0


def word_error_rate_calculator(ground_truth, predictions):
    if not ground_truth:
        return 0.0
    errors = sum(gt != pred for gt, pred in zip(ground_truth, predictions))
    return errors / len(ground_truth)


def character_accuracy_calculator(ground_truth, predictions):
    return 1 - character_error_rate_calculator(ground_truth, predictions)


def word_accuracy_calculator(ground_truth, predictions):
    return 1 - word_error_rate_calculator(ground_truth, predictions)


def __levenshtein_distance(seq1, seq2):
    previous_row = list(range(len(seq2) + 1))
    for i, item1 in enumerate(seq1):
        current_row = [i + 1]
        for j, item2 in enumerate(seq2):
            insertions = current_row[j] + 1
            deletions = previous_row[j + 1] + 1
            substitutions = previous_row[j] + (item1 != item2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
