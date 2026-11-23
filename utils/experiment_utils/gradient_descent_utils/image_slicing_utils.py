import torch

from info.vocab_info import SOS_CHARACTER
from utils.character_index_converter import letter_to_index_converter


def get_one_hot_start_token(batch_size, vocab_length):
    start_token = torch.zeros((batch_size, vocab_length), dtype=torch.double)
    start_token[:, letter_to_index_converter[SOS_CHARACTER]] = 1
    return start_token


def get_start_token(batch_size):
    return torch.full((batch_size, 1), letter_to_index_converter[SOS_CHARACTER], dtype=torch.long)


def ctc_clean(letters_indexes_list):
    pass

# def clean_ctc(s):
#     # remove repeated
#     word = ""
#     prev = ""
#     for c in s:
#         if c != prev:
#             word += c
#         prev = c
#     word = re.sub("[\-SE]", "", word)
#     return word