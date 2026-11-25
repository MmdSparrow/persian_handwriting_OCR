import torch

from info.vocab_info import EOS_CHARACTER, PADDING_CHARACTER, SOS_CHARACTER, VOCAB_INDEXES_LETTERS_DICT, VOCAB_LETTERS_INDEXES_DICT

__all__ = [
    "clean_encoded_labels",
    "index_to_letter_converter",
    "letter_to_index_converter",
]


def letter_to_index_converter(word_str):
    return torch.tensor([VOCAB_LETTERS_INDEXES_DICT[character] for character in word_str], dtype=torch.long)


def index_to_letter_converter(word_index):
    return "".join(VOCAB_INDEXES_LETTERS_DICT[index] for index in word_index)


def index_to_letter_converter_batch_version(encoded_words_batch):
    return [index_to_letter_converter(word_index) for word_index in encoded_words_batch]


def clean_encoded_labels(encoded_prediction_batch):
    sos_index = VOCAB_LETTERS_INDEXES_DICT[SOS_CHARACTER]
    pad_index = VOCAB_LETTERS_INDEXES_DICT[PADDING_CHARACTER]
    eos_index = VOCAB_LETTERS_INDEXES_DICT[EOS_CHARACTER]
    cleaned_batch = []
    for prediction in encoded_prediction_batch:
        cleaned_prediction = []
        for token in prediction:
            token = int(token)
            if token == eos_index:
                break
            if token in (sos_index, pad_index):
                continue
            cleaned_prediction.append(token)
        cleaned_batch.append(cleaned_prediction)
    return cleaned_batch


def clean_encoded_labels_batch(labels_):
    pass
