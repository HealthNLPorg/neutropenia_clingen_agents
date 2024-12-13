import argparse
import logging
import os
import pathlib
from typing import Iterable, Optional

from tika import parser

argparser = argparse.ArgumentParser(description="")
argparser.add_argument("--input_file", type=str)
argparser.add_argument("--input_dir", type=str)
argparser.add_argument("--output_dir", type=str)
argparser.add_argument("--already_processed", type=str)
argparser.add_argument("--timeout_cases", type=str)
argparser.add_argument("--tika_timeout", type=int, default=120)
# Might switch things around soon
argparser.add_argument("--TCGA", action="store_true")


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
OUTPUT_EXT = ".txt"

TIMED_OUT_FILES = "./timeouts.txt"
TikaParse = dict[str, dict[str, str | list[str]] | str | list[str]]


def get_files(raw_dir: str, already_processed: str | None, TCGA: bool) -> Iterable[str]:
    def to_process(fn: str) -> bool:
        return True

    if already_processed is not None:
        with open(already_processed, mode="rt") as f:
            processed_file_stems = {fn.strip() for fn in f.readlines()}

        if TCGA:

            def to_process(fn: str) -> bool:
                return fn not in processed_file_stems

        else:

            def to_process(fn: str) -> bool:
                return pathlib.Path(fn).stem.strip() not in processed_file_stems

    if TCGA:
        for root, _, files in os.walk(raw_dir):
            for fn in files:
                if fn.lower().endswith("pdf"):
                    yield os.path.join(root, fn)
    else:
        for fn in os.listdir(raw_dir):
            if to_process(fn):
                yield os.path.join(raw_dir, fn)


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def get_inputs(
    input_file: str | None,
    input_dir: str | None,
    already_processed: str | None,
    TCGA: bool,
) -> Iterable[str]:
    if input_file is not None:
        return [input_file]
    elif input_dir is not None:
        return get_files(input_dir, already_processed, TCGA)
    else:
        logger.warning("No input provided")
        return []


def get_parses(
    input_file: str | None,
    input_dir: str | None,
    already_processed: str | None,
    timeout_cases: str | None,
    timeout: int,
    TCGA: bool,
) -> Iterable[tuple[TikaParse, str]]:
    for fn in get_inputs(input_file, input_dir, already_processed, TCGA):
        tika_parse = parse_pdf(fn, timeout, timeout_cases)
        if tika_parse is not None:
            yield (tika_parse, fn)


def ocr_used(parse_result: TikaParse) -> bool:
    metadata = parse_result.get("metadata", {})
    parser_trace: list[str] = []
    if isinstance(metadata, dict):
        try_parser_trace = metadata.get("X-TIKA:Parsed-By-Full-Set")
        if not isinstance(try_parser_trace, list):
            raise ValueError(f"Malformed Tika Parse or no metadata {parse_result}")
        parser_trace = try_parser_trace
    else:
        raise ValueError(f"Malformed Tika Parse or no metadata {parse_result}")
    try:
        return "org.apache.tika.parser.ocr.TesseractOCRParser" in parser_trace
    except Exception as e:
        raise ValueError(
            f"{parser_trace} missing annotator trajectory {parse_result}"
        ) from e


def parse_pdf(fn: str, timeout: int, timeout_cases: str | None) -> Optional[TikaParse]:
    headers = {
        "X-Tika-OCRLanguage": "eng",
        # "X-Tika-PDFextractInlineImages": "true",
    }
    logger.info(f"processing {fn}")
    try:
        parsed_pdf = parser.from_file(
            fn, requestOptions={"headers": headers, "timeout": timeout}
        )
    except Exception:
        logger.info(f"{fn} timed out")
        if timeout_cases is not None:
            with open(timeout_cases, mode="at") as timeout_f:
                timeout_f.write(f"{fn}\n")
        return None
    logger.info(f"processed {fn}")
    return parsed_pdf


def process_tcga(
    input_dir: str,
    already_processed: str,
    timeout_cases: str,
    tika_timeout: int,
) -> None:
    def process_output(ocr_used: bool, content: str, fn: str, output_dir: str) -> None:
        ocr_meta = "OCR" if ocr_used else "NATIVE"
        out_fn = os.path.join(
            output_dir, f"{ocr_meta}_{basename_no_ext(fn)}{OUTPUT_EXT}"
        )
        with open(out_fn, mode="wt") as out_writer:
            out_writer.write(content)

    for parsed_pdf, fn in get_parses(
        input_file=None,
        input_dir=input_dir,
        already_processed=already_processed,
        timeout_cases=timeout_cases,
        timeout=tika_timeout,
        TCGA=True,
    ):

        # parsed_pdf = parse_pdf(fn)
        ocr_parsed = ocr_used(parsed_pdf)
        pdf_content = parsed_pdf.get("content")
        if pdf_content is not None and isinstance(pdf_content, str):
            process_output(
                ocr_parsed, pdf_content.strip(), fn, str(pathlib.Path(fn).parent)
            )
            with open(already_processed, mode="at") as f:
                f.write(f"{fn}\n")
        elif pdf_content is None:
            raise ValueError(
                f"Content for {fn} is {pdf_content}, possible issue with Tesseract OCR not being invoked"
            )
        else:
            raise TypeError(
                f"Content for {fn} should be either None or str, is {type(pdf_content)}"
            )


def process(
    input_file: str,
    input_dir: str,
    output_dir: str | None,
    already_processed: str,
    timeout_cases: str,
    tika_timeout: int,
    TCGA: bool,
) -> None:
    if (
        TCGA
    ):  # Not the way I prefer to do things but might have the chance to refactor later
        print("Present day")
        process_tcga(
            input_dir,
            already_processed,
            timeout_cases,
            tika_timeout,
        )
        print("Present time")
        return

    def process_output(ocr_used: bool, content: str, fn: str) -> None:
        pass

    if output_dir is not None and not TCGA:

        def process_output(ocr_used: bool, content: str, fn: str) -> None:
            ocr_meta = "OCR" if ocr_used else "NATIVE"
            out_fn = os.path.join(
                output_dir, f"{ocr_meta}_{basename_no_ext(fn)}{OUTPUT_EXT}"
            )
            with open(out_fn, mode="wt") as out_writer:
                out_writer.write(content)

    else:

        def process_output(ocr_used: bool, content: str, fn: str) -> None:
            print(content)

    for parsed_pdf, fn in get_parses(
        input_file=input_file,
        input_dir=input_dir,
        already_processed=already_processed,
        timeout_cases=timeout_cases,
        timeout=tika_timeout,
        TCGA=False,
    ):
        # parsed_pdf = parse_pdf(fn)
        ocr_parsed = ocr_used(parsed_pdf)
        pdf_content = parsed_pdf.get("content")
        if pdf_content is not None and isinstance(pdf_content, str):
            process_output(ocr_parsed, pdf_content.strip(), fn)
            with open(already_processed, mode="at") as f:
                f.write(f"{pathlib.Path(fn).stem}\n")
        elif pdf_content is None:
            raise ValueError(
                f"Content for {fn} is {pdf_content}, possible issue with Tesseract OCR not being invoked"
            )
        else:
            raise TypeError(
                f"Content for {fn} should be either None or str, is {type(pdf_content)}"
            )


def main() -> None:
    args = argparser.parse_args()
    print(args)
    process(
        args.input_file,
        args.input_dir,
        args.output_dir,
        args.already_processed,
        args.timeout_cases,
        args.tika_timeout,
        args.TCGA,
    )


if __name__ == "__main__":
    main()
