from info.vocab_info import VOCAB_INDEXES_LETTERS_DICT, VOCAB_LETTERS_INDEXES_DICT

__all__ = [
    "index_to_letter_converter",
    "letter_to_index_converter",
]


def letter_to_index_converter(word_str):
    return [(VOCAB_LETTERS_INDEXES_DICT[character]) for character in word_str]


def index_to_letter_converter(word_index):
    return [(VOCAB_INDEXES_LETTERS_DICT[index]) for index in word_index]
