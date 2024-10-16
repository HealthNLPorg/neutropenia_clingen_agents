from tika import parser
import argparse
import os
import logging
import pathlib
from typing import Iterable


argparser = argparse.ArgumentParser(description="")
argparser.add_argument("--input_file", type=str)
argparser.add_argument("--input_dir", type=str)
argparser.add_argument("--output_dir", type=str)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
OUTPUT_EXT = ".txt"


def get_files(raw_dir: str) -> Iterable[str]:
    for base_fn in os.listdir(raw_dir):
        yield os.path.join(raw_dir, base_fn)


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def get_inputs(input_file: str, input_dir: str) -> Iterable[str]:
    if input_file is not None:
        return [input_file]
    elif input_dir is not None:
        return get_files(input_dir)
    else:
        logger.warning("No input provided")
        return []


def get_pdf_content(fn: str) -> str:
    parsed_pdf = parser.from_file(fn)
    return parsed_pdf["content"].strip()


def main() -> None:
    args = argparser.parse_args()

    def process_output(content: str) -> None:
        pass

    if args.output_dir is not None:

        def process_output(content: str) -> None:
            out_fn = os.path.join(args.output_dir, f"{basename_no_ext(fn)}{OUTPUT_EXT}")
            with open(out_fn, mode="wt") as out_writer:
                out_writer.write(content)

    else:

        def process_output(content: str) -> None:
            print(content)

    for fn in get_inputs(args.input_file, args.input_dir):
        pdf_content = get_pdf_content(fn)
        process_output(pdf_content)


if __name__ == "__main__":
    main()
