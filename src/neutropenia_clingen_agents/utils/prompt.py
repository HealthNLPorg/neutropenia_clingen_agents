import pathlib
import re
from collections.abc import Callable, Iterable
from itertools import chain
from typing import cast

import polars as pl

from .serialization import reinsert_whitespace

Message = dict[str, str]


def few_shot_prompt(
    system_prompt: str, query: str, examples: Iterable[tuple[str, str]]
) -> list[Message]:
    def message_pair(ex_query: str, ex_answer: str) -> tuple[Message, ...]:
        return {"role": "user", "content": ex_query}, {
            "role": "assistant",
            "content": ex_answer,
        }

    few_shot_examples = chain.from_iterable(
        message_pair(ex_query=ex_query, ex_answer=ex_answer)
        for ex_query, ex_answer in examples
    )

    messages = [
        {"role": "system", "content": system_prompt},
        *few_shot_examples,
        {"role": "user", "content": query},
    ]
    return messages


def get_huggingface_prompt_builder(
    examples_file: str | None, sample_document: str | None, sample_answer: str | None
) -> Callable[[str, str], list[Message]]:
    def few_shot_with_examples(
        examples: Iterable[tuple[str, str]],
    ) -> Callable[[str, str], list[Message]]:
        def _few_shot_prompt(s, q):
            return few_shot_prompt(system_prompt=s, query=q, examples=examples)

        return _few_shot_prompt

    if examples_file is not None:
        examples = get_examples(examples_file)
        if len(examples) > 0:
            return few_shot_with_examples(examples=examples)

        else:
            ValueError("Empty examples file")

            return empty_prompt
    elif sample_document is not None and sample_answer is not None:
        example = get_document_level_example(sample_document, sample_answer)
        if all(len(ex) > 0 for ex in example):
            return few_shot_with_examples(examples=(example,))
        else:
            ValueError("Empty sample document and/or empty sample answer")

            return empty_prompt
    else:
        return zero_shot_prompt


def get_langchain_examples(examples_file_path: str) -> list[dict[str, str]]:
    return [
        {"input": _input, "output": _output}
        for _input, _output in get_examples(examples_file_path)
    ]


def get_examples(examples_file_path: str) -> list[tuple[str, str]]:
    suffix = pathlib.Path(examples_file_path).suffix.lower()
    match suffix.strip():
        case ".tsv":
            full_dataframe = pl.read_csv(examples_file_path, separator="\t")
            raw_queries = cast(
                Iterable[str],
                (
                    full_dataframe["query"]
                    if "query" in full_dataframe.columns
                    else full_dataframe["sentence"]
                ),
            )
            queries = (reinsert_whitespace(query) for query in raw_queries)
            responses = cast(Iterable[str], full_dataframe["response"])
            examples = list(zip(queries, responses))
        case ".txt" | "":
            examples = parse_input_output(examples_file_path)
        case _:
            ValueError(f"Presently unsupported examples file format {suffix}")
            examples = []
    return examples


def parse_input_output(examples_file_path: str) -> list[tuple[str, str]]:
    def parse_example(raw_example: str) -> tuple[str, str]:
        result = tuple(
            elem.strip()
            for elem in re.split("input:|output:", raw_example)
            if len(elem.strip()) > 0
        )
        assert len(result) == 2
        return result

    with open(examples_file_path, encoding="utf-8") as ef:
        raw_str = ef.read()
        return [
            parse_example(example.strip())
            for example in raw_str.split("\n\n")
            if len(example.split()) > 0
        ]


def get_document_level_example(
    sample_document_path: str, sample_answer_path: str
) -> tuple[str, str]:
    with open(sample_document_path, encoding="utf-8") as sample_document:
        # not normalizing newlines since those might be useful
        query = sample_document.read()
    sample_answer_dataframe = pl.read_csv(sample_answer_path, separator="\t")
    # specific to earlier use-case etc but for now
    answer = "\n".join(cast(Iterable[str], sample_answer_dataframe["query"]))
    return (query, answer)


def zero_shot_prompt(system_prompt: str, query: str) -> list[Message]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return messages


def empty_prompt(system_prompt: str, query: str) -> list[Message]:
    return []
