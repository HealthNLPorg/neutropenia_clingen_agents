# retain newline information via special markers
# while removing them for storage
# ( so you can load them later via pandas without parsing errors )
import string
import unicodedata
from functools import cache


def clean_whitespace(sample: str) -> str:
    return (
        sample.replace("\n", "<cn>")
        .replace("\t", "<ct>")
        .replace("\f", "<cf>")
        .replace("\r", "<cr>")
    )


def reinsert_whitespace(sample: str) -> str:
    return (
        sample.replace("<cn>", "\n")
        .replace("<ct>", "\t")
        .replace("<cf>", "\f")
        .replace("<cr>", "\r")
    )


@cache
def relevant_unicode_category(category: str) -> bool:
    return category != "So" and not category.startswith("C")


@cache
def relevant_character(char: str) -> bool:
    if char in string.printable:
        return True
    category = unicodedata.category(char)
    return relevant_unicode_category(category)


def aggresive_normalize(sample: str) -> str:
    return " ".join("".join(filter(relevant_character, sample)).split())
