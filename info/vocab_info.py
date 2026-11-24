PERSIAN_ALPHABET_LETTERS = [
    "آ",
    "ا",
    "ب",
    "پ",
    "ت",
    "ث",
    "ج",
    "چ",
    "ح",
    "خ",
    "د",
    "ذ",
    "ر",
    "ز",
    "ژ",
    "س",
    "ش",
    "ص",
    "ض",
    "ط",
    "ظ",
    "ع",
    "غ",
    "ف",
    "ق",
    "ک",
    "گ",
    "ل",
    "م",
    "ن",
    "و",
    "ه",
    "ی",
    "ئ",
]

SOS_CHARACTER = "S"
EOS_CHARACTER = "E"
PADDING_CHARACTER = "P"
BLANK_CHARACTER = "-"
SPACE_CHARACTER = " "

SPECIAL_CHARACTERS = [
    SOS_CHARACTER,
    EOS_CHARACTER,
    SPACE_CHARACTER,
]
__VOCAB_LETTERS = SPECIAL_CHARACTERS + PERSIAN_ALPHABET_LETTERS

VOCAB_LETTERS_INDEXES_DICT = {letter: index for index, letter in enumerate(__VOCAB_LETTERS)}
VOCAB_INDEXES_LETTERS_DICT = {index: letter for index, letter in enumerate(__VOCAB_LETTERS)}
