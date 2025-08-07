import json
from typing import Iterable, Tuple

import pandas as pd


def process() -> None:
    df = pd.DataFrame(
        parse_and_stream(
            "./llama3_step1_results.tsv", "./syntax_statement_parse_few_shot.tsv"
        ),
        columns=[
            "Gene",
            "Statement",
            "Nucleotide Syntax",
            "Protein Syntax",
            "Text Sample",
        ],
    )
    # may be unnecessary so leaving out for now
    # df.reset_index(drop=True)
    df.to_excel("./example_1.xlsx", index=False)


def parse_and_stream(queries_fn: str, answers_fn: str) -> Iterable[Tuple[str, ...]]:
    queries_df = pd.read_csv(queries_fn, sep="\t")
    answers_df = pd.read_csv(answers_fn, sep="\t")
    for query, raw_answer in zip(queries_df["query"], answers_df["answers"]):
        answer = raw_answer.split("assistant")[-1].strip()
        try:
            results = json.loads(answer)
            yield (
                " ".join(results["GENE"]),
                " ".join(results["STATEMENT"]),
                " ".join(results["SYNTAX_N"]),
                " ".join(results["SYNTAX_P"]),
                query,
            )
        except Exception:
            print(f"No results parsed from {raw_answer}")


def main() -> None:
    process()


if __name__ == "__main__":
    main()
