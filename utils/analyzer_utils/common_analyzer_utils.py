__all__ = [
    "write_title_in_file",
]


def __generate_title(title):
    TOTAL_LENGTH = 130
    line = "=" * ((TOTAL_LENGTH - len(title)) // 2)
    return f"{line} {title.upper()} {line}\n"


def write_title_in_file(title, file_path):
    str_to_write = __generate_title(title)
    with open(file_path, "a") as file_to_write:
        file_to_write.write(f"{str_to_write}")
