import argparse
import logging
import os
import pathlib
from collections.abc import Iterable
from time import time

from datasets import load_dataset

from .gene_mention_agent.core import Agent
from .utils.prompt import get_langchain_examples

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
    "--max_length",
    type=int,
    default=8_000,
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


logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def process(
    query_tsv: str,
    output_dir: str,
    model_id: str,
    examples_file: str,
    sample_document: str,
    sample_answer: str,
    prompt_file: str,
    max_new_tokens: int,
    max_length: int,
    batch_size: int,
) -> None:
    query_dataset = load_dataset(
        "csv",
        sep="\t",
        data_files=(query_tsv,),
    )
    query_dataset = query_dataset["train"]
    logger.info(f"Dataset loaded from {query_tsv}")

    out_dir = output_dir
    out_fn_stem = pathlib.Path(query_tsv).stem
    tsv_out_fn = f"{out_fn_stem}.tsv"
    tsv_out_path = os.path.join(out_dir, tsv_out_fn)

    with open(prompt_file, encoding="utf-8") as f:
        system_prompt = f.read()

    start = time()
    gene_mention_agent = Agent(
        system_prompt=system_prompt,
        examples=get_langchain_examples(examples_file),
        model_id=model_id,
        model_kwargs={"max_length": max_length},
        pipeline_kwargs={"max_new_tokens": max_new_tokens},
    )

    end = time()
    logger.info(f"Loading model took {end - start} seconds")

    def format_instance(batch):
        batch["input"] = [{"input": sentence} for sentence in batch["sentence"]]
        return batch

    def predict(batch):
        # Can't believe I'm putting
        # a branch in a function like this but
        # Huggingface has screwed up one too many times
        try:
            batch["output"] = gene_mention_agent(batch["input"])
        except Exception:
            logger.warning("Ran into issue processing the following batch")
            logger.warning(batch)
        return batch

    # def clean_output(batch):
    #     logger.info(batch["output"])
    #     batch["output"] = [instance.get("AI", "ERROR") for instance in batch["output"]]
    #     return batch

    query_dataset = (
        query_dataset.map(format_instance, batched=True, batch_size=batch_size).map(
            predict, batched=True, batch_size=batch_size
        )
        # .map(clean_output, batched=True, batch_size=batch_size)
    )
    query_dataframe = query_dataset.to_polars()
    query_dataframe.write_csv(tsv_out_path, separator="\t")


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
        max_length=args.max_length,
        batch_size=args.batch_size,
    )


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def get_files(raw_dir: str) -> Iterable[str]:
    for base_fn in os.listdir(raw_dir):
        yield os.path.join(raw_dir, base_fn)


if __name__ == "__main__":
    main()
