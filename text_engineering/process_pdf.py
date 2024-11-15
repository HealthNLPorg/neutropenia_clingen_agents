import argparse
import logging
import os
import pathlib
# import subprocess
from typing import Iterable, Optional

from tika import parser

argparser = argparse.ArgumentParser(description="")
argparser.add_argument("--input_file", type=str)
argparser.add_argument("--input_dir", type=str)
argparser.add_argument("--output_dir", type=str)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
OUTPUT_EXT = ".txt"


def get_files(raw_dir: str) -> Iterable[str]:
    for base_fn in os.listdir(raw_dir):
        yield os.path.join(raw_dir, base_fn)


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def get_inputs(input_file: str | None, input_dir: str | None) -> Iterable[str]:
    if input_file is not None:
        return [input_file]
    elif input_dir is not None:
        return get_files(input_dir)
    else:
        logger.warning("No input provided")
        return []


def get_pdf_content(fn: str) -> Optional[str]:
    headers = {
        "X-Tika-OCRLanguage": "eng",
        # "X-Tika-PDFextractInlineImages": "true",
    }
    print(f"processing {fn}")
    parsed_pdf = parser.from_file(fn)
    # , headers=headers)
    # , headers=headers)
    content = parsed_pdf.get("content")
    # https://stackoverflow.com/a/48153647
    # convert to temporary image file via
    # ImageMagick and thence
    # Tika hoping it uses Tesseract appropriately
    if content is None:
        print("Trying ocr parse")
        # stem = pathlib.Path(fn).stem

        logger.warning(f"No embedded text found in {fn}, resorting to Tesseract OCR")
        # Run ImageMagick via subprocess (command line)
        # temp_tiff = f"~/.tmp_{stem}.tiff"
        # params = [
        #     "convert",
        #     fn,
        #     "-density",
        #     "300",  # u, whatever this was
        #     "-depth",
        #     "8",
        #     "-strip",
        #     "-background",
        #     "white",
        #     "-alpha",
        #     "off",
        #     temp_tiff,
        # ]
        # subprocess.check_call(params)

        # # Run Tika again on new temp.tiff file
        # parsed_tiff = parser.from_file(temp_tiff, headers=headers)
        ocr_parsed_pdf = parser.from_file(fn, headers=headers)

        # Delete the temporary file
        # os.remove(temp_tiff)
        # content = parsed_tiff.get("content")
        content = ocr_parsed_pdf.get("content")
        if content is None:
            raise ValueError("Tried everything so far")
    print(f"processed {fn}")
    return content.strip()
    # , results {parsed_pdf}")


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
        if pdf_content is not None:
            process_output(pdf_content)


if __name__ == "__main__":
    main()
