import pandas as pd
import os
import argparse
import logging
from functools import partial


argparser = argparse.ArgumentParser(description="")
argparser.add_argument("--input_file", type=str)
argparser.add_argument("--output_dir", type=str)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

DELIMITER_ESCAPE = {"\t": "<ct>"}


def fix_row(
    total_columns: int, frontwards_correct: bool, delimiter: str, row: str
) -> list[str]:
    cells = row.split(delimiter)
    if len(cells) == total_columns:
        return cells
    elif len(cells) < total_columns:
        logger.warning(f"Fewer cells than column headers {cells}")
        exit(1)
        return []
    if frontwards_correct:
        correct_cells = cells[:total_columns]
        incorrect_cells = cells[total_columns:]
        return [*correct_cells, DELIMITER_ESCAPE[delimiter].join(incorrect_cells)]
    else:
        reordered_cells = cells[::-1]
        correct_cells = reordered_cells[:total_columns]
        incorrect_cells = reordered_cells[total_columns:]
        return [
            DELIMITER_ESCAPE[delimiter].join(incorrect_cells[::-1]),
            *correct_cells[::-1],
        ]


def fix_frame(
    input_file: str,
    output_dir: str,
    delimiter: str = "\t",
    frontwards_correct: bool = False,
) -> None:
    # Assume columns are correct
    def clean(s: str)-> str:
        return s.removesuffix("\n")

    with open(input_file, mode="r") as f:
        flines = map(clean, f.readlines())
    # logger.info(f"Total instances {len(flines)}")
    # raw_column_headers = flines[0]
    raw_column_headers = next(flines)
    column_headers = raw_column_headers.split(delimiter)
    print(column_headers)
    local_fix_row = partial(fix_row, len(column_headers), frontwards_correct, delimiter)
    stuff = list(map(local_fix_row, flines))
    print(stuff[:5])
    output_frame = pd.DataFrame(stuff, columns=column_headers)
    output_frame.to_csv(
        os.path.join(output_dir, os.path.basename(input_file)), sep="\t", index=False
    )


def main() -> None:
    args = argparser.parse_args()
    fix_frame(args.input_file, args.output_dir)


if __name__ == "__main__":
    main()
