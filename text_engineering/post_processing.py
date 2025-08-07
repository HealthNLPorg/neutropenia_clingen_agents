import argparse
import os
import pandas as pd
from functools import partial
from typing import Callable, Any
import logging


parser = argparse.ArgumentParser(description="")

parser.add_argument("--input_file", type=str)
parser.add_argument("--output_dir", type=str)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

ATTRIBUTE_COLUMNS = ["Syntax_N", "Syntax_P", "Type", "Vaf"]
MENTION_COLUMNS = ["Gene", *ATTRIBUTE_COLUMNS]
COLUMNS = [
    *MENTION_COLUMNS,
    "Sentence",
    "Section",
    "Specimen_Collection_Date",
    "Sample_Source",
    "Filename",
]


def filter_hallucinations(df: pd.DataFrame) -> pd.DataFrame:
    def __normalize(sample: Any) -> str:
        return str(sample).strip().lower()

    def unkify_hallucination(
        prediction_key: str,
        reference_key: str,
        normalize: Callable[[str], str],
        row: pd.Series,
    ) -> str:
        normalized_prediction = normalize(row[prediction_key])
        if (
            len(normalized_prediction) > 0
            and normalized_prediction != "__UNK__"
            and normalized_prediction in normalize(row[reference_key])
        ):
            return str(row[prediction_key]).strip()
        return "__UNK__"

    for mention_column in MENTION_COLUMNS:
        local_unkify_hallucination = partial(
            unkify_hallucination, mention_column, "Sentence", __normalize
        )
        df[mention_column] = df.apply(local_unkify_hallucination, axis=1)
    return df


def filter_empty_mentions(df: pd.DataFrame) -> pd.DataFrame:
    def sample_populated(row: str) -> str:
        return row["Gene"] != "__UNK__" and any(
            row[attribute_column] != "__UNK__" for attribute_column in ATTRIBUTE_COLUMNS
        )

    return df[df.apply(sample_populated, axis=1)]


def post_process(input_file: str, output_dir: str) -> None:
    df = pd.read_csv(input_file, sep="\t")
    df = df.fillna("__UNK__")
    halluccinations_filtered_df = filter_hallucinations(df)
    empties_filtered_df = filter_empty_mentions(halluccinations_filtered_df)
    empties_filtered_df.to_csv(
        os.path.join(output_dir, f"post_processed_{os.path.basename(input_file)}"),
        sep="\t",
        index=False,
    )


def main() -> None:
    args = parser.parse_args()
    post_process(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
