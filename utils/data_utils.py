from info.vocab_info import PADDING_CHARACTER

__all__ = [
    "padding_labels",
]


def padding_labels(labels_list, maximum_length):
    return [s.ljust(maximum_length, PADDING_CHARACTER) for s in labels_list]
    # return [s.rjust(maximum_length, PADDING_CHARACTER) for s in labels_list]
