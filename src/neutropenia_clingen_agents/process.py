import argparse
import json
import logging
import os
import pathlib
from collections.abc import Iterable, Mapping, MutableMapping
from functools import partial
from time import time
from typing import Any, cast

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)

from .post_process import post_process_dataset
from .utils.filesystem import make_directory
from .utils.prompt import get_huggingface_prompt_builder

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--examples_file",
    type=str,
)
parser.add_argument(
    "--sample_document",
    type=str,
)
parser.add_argument(
    "--sample_answer",
    type=str,
)
parser.add_argument("--prompt_file", type=str)
parser.add_argument(
    "--model_id",
    type=str,
)


parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=512,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=4,
)
parser.add_argument(
    "--query_tsv",
    type=str,
)

parser.add_argument("--output_dir", type=str)
parser.add_argument("--post_process", action="store_true")


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_raw_output(sample: MutableMapping[str, Any]) -> Mapping[str, Any]:
    sample["raw_output"] = json.dumps(
        sample["output"][0]["generated_text"].split("assistant")[-1].strip()
    )
    sample["output"] = str(sample["output"])
    return sample


def process(
    query_tsv: str,
    output_dir: str,
    model_id: str,
    examples_file: str,
    sample_document: str,
    sample_answer: str,
    prompt_file: str,
    max_new_tokens: int,
    batch_size: int,
    post_process: bool,
) -> None:
    query_dataset = load_dataset(
        "csv",
        sep="\t",
        data_files=(query_tsv,),
    )
    query_dataset = query_dataset["train"]
    logger.info(f"Dataset loaded from {query_tsv}")

    make_directory(output_dir)
    query_tsv_stem = pathlib.Path(query_tsv).stem

    with open(prompt_file, encoding="utf-8") as f:
        system_prompt = f.read()

    build_huggingface_prompt = get_huggingface_prompt_builder(
        examples_file, sample_document, sample_answer
    )
    start = time()
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer is None:
        raise ValueError(f"Could not load tokenizer for {model_id}")
    # device_map="auto" problematic on E3 - as of July 2026
    # the Auto...from_pretrained... then pipeline(...) with no config
    # seems to be the best idiom for E3
    seqgen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # HF increasingly pushing this
        token=os.environ["HF_TOKEN"],
    )

    end = time()
    logger.info(f"Loading model took {end - start} seconds")

    local_build_prompt = partial(build_huggingface_prompt, system_prompt)

    def __apply_chat_template(prompt: list[dict[str, str]]) -> str:
        try:
            return cast(
                str,
                # seqgen_pipe.tokenizer.apply_chat_template(
                tokenizer.apply_chat_template(
                    prompt,
                    tokenize=False,
                    add_generation_prompt=False,
                    truncate=True,
                    # max_new_tokens is the new kwarg for max_length
                    max_new_tokens=max_new_tokens,
                ),
            )
        except Exception as e:
            raise ValueError(f"Ran into {e}")

    def format_to_chat_template(
        sample: MutableMapping[str, Any],
    ) -> MutableMapping[str, Any]:
        prompts = map(local_build_prompt, sample["sentence"])
        sample["text"] = [__apply_chat_template(prompt) for prompt in prompts]
        return sample

    def predict(sample: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        try:
            sample["output"] = seqgen_pipe(sample["text"])
        except Exception:
            logger.warning("Ran into issue processing the following sample")
            logger.warning(sample)
        return sample

    processed_query_dataset = (
        query_dataset.map(format_to_chat_template, batched=True, batch_size=batch_size)
        .map(predict, batched=True, batch_size=batch_size)
        .map(parse_raw_output)
    )
    # processed_query_json = f"processed_{query_tsv_stem}.json"
    # processed_json_out_path = os.path.join(output_dir, processed_query_json)
    # processed_query_dataset.to_json(processed_json_out_path)
    processed_query_tsv = f"processed_{query_tsv_stem}.tsv"
    processed_tsv_out_path = os.path.join(output_dir, processed_query_tsv)
    query_dataframe = processed_query_dataset.to_polars()
    query_dataframe.write_csv(processed_tsv_out_path, separator="\t")
    if post_process:
        post_processed_tsv_query_tsv = f"post_processed_{query_tsv_stem}.tsv"
        post_processed_tsv_out_path = os.path.join(
            output_dir, post_processed_tsv_query_tsv
        )
        post_process_dataset(processed_query_dataset, post_processed_tsv_out_path)


def main() -> None:
    args = parser.parse_args()
    process(
        query_tsv=args.query_tsv,
        output_dir=args.output_dir,
        model_id=args.model_id,
        examples_file=args.examples_file,
        sample_document=args.sample_document,
        sample_answer=args.sample_answer,
        prompt_file=args.prompt_file,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        post_process=args.post_process,
    )


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def get_files(raw_dir: str) -> Iterable[str]:
    for base_fn in os.listdir(raw_dir):
        yield os.path.join(raw_dir, base_fn)


if __name__ == "__main__":
    main()
