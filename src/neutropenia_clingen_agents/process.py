import argparse
import json
import logging
import os
import pathlib
from collections.abc import Iterable
from functools import partial
from time import time

from datasets import load_dataset
from transformers import pipeline

from .utils.prompt import build_prompt_template

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--examples_file",
    type=str,
    help="Check the `get_examples` method for the possible formats for now",
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
    "--model_path",
    type=str,
)


parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--fancy_output", action="store_true")
parser.add_argument("--model_name", choices=["llama2", "llama3", "mixtral", "qwen2"])


parser.add_argument(
    "--max_new_tokens",
    type=int,
)
parser.add_argument(
    "--batch_size",
    type=int,
)
parser.add_argument(
    "--query_files",
    nargs="+",
    default=[],
    help="TSVs for now",
)

parser.add_argument(
    "--query_dir",
    help="TSVs for now",
)
parser.add_argument("--output_dir", type=str)

name2path = {
    "llama2": "/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
    "llama3": "/lab-share/CHIP-Savova-e2/Public/resources/Meta-Llama-3-8B-Instruct/",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen2": "Qwen/Qwen2-1.5B-Instruct",
}

ATTRIBUTES = {"VAF", "SYNTAX_N", "SYNTAX_P", "TYPE"}
# {role: {system|user|assistant}, content: ...}

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


# class PooledDataset(TorchDataset):
#     def __init__(self, device_index: int, world_size: int, original_dataset) -> None:
#         super().__init__()
#         self.device_index = device_index
#         self.data = original_dataset[device_index::world_size]
#         logger.info(self.data)

#     def __len__(self) -> int:
#         return len(self.data)

#     def __getitem__(self, i):
#         return self.data[i]


def main() -> None:
    args = parser.parse_args()
    final_path = ""
    if args.model_name is not None:
        final_path = name2path[args.model_name]
    else:
        final_path = args.model_path
    logger.info(f"Loading tokenizer and model for model name {final_path}")
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=args.load_in_4bit, load_in_8bit=args.load_in_8bit
    # )
    logger.info("Building dataset")
    query_dataset = load_dataset(
        "csv",
        sep="\t",
        data_files=os.listdir(args.query_dir) if args.query_dir else args.query_files,
    )
    query_dataset = query_dataset["train"]

    start = time()
    seqgen_pipe = pipeline(
        "text-generation",
        model=final_path,
        device_map="auto",
        max_new_tokens=args.max_new_tokens,
    )
    end = time()
    logger.info(f"Loading model took {end - start} seconds")
    out_dir = args.output_dir
    out_fn_stem = pathlib.Path(
        args.query_dir
        if args.query_dir
        else "_".join(basename_no_ext(fn) for fn in args.query_files)
    ).stem
    tsv_out_fn = f"{out_fn_stem}.tsv"
    tsv_out_path = os.path.join(out_dir, tsv_out_fn)

    prompt_template = build_prompt_template(
        args.examples_file, args.sample_document, args.sample_answer
    )

    with open(args.prompt_file, encoding="utf-8") as f:
        system_prompt = f.read()
    prompt_wrapper = partial(prompt_template, system_prompt)

    def format_chat(sample: dict) -> dict:
        return {
            "text": seqgen_pipe.tokenizer.apply_chat_template(
                prompt_wrapper(sample["sentence"]),
                tokenize=False,
                add_generation_prompt=False,
                truncate=True,
                max_length=args.max_length,
            )
        }

    def predict(batch):
        # Can't believe I'm putting
        # a branch in a function like this but
        # Huggingface has screwed up one too many times
        try:
            batch["output"] = seqgen_pipe(batch["text"])
        except Exception:
            logger.warning("Ran into issue processing the following batch")
            logger.warning(batch)
        return batch

    query_dataset = (
        query_dataset.map(format_chat)
        .map(predict, batched=True, batch_size=args.batch_size)
        .map(parse_output)
        .map(insert_mentions)
        .map(clean_section)
        .remove_columns(["text", "output", "json_output"])
    )
    query_dataframe = query_dataset.with_format("polars")
    renamed_column_mapping = {col: col.title() for col in query_dataframe.columns}
    query_dataframe = query_dataframe.rename(mapping=renamed_column_mapping)
    query_dataframe = query_dataframe[
        [
            "Gene",
            *sorted(map(str.title, ATTRIBUTES)),
            "Sentence",
            "Section",
            "Specimen_Collection_Date",
            "Sample_Source",
            "Filename",
        ]
    ]
    query_dataframe.write_csv(tsv_out_path, sep="\t")


def try_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def non_empty_json(sample: dict) -> bool:
    return len(sample["json_output"]) > 0


def parse_output(sample: dict) -> dict:
    model_answer = sample["output"][0]["generated_text"].split("assistant")[-1].strip()
    sample["json_output"] = json.dumps(try_json(model_answer))
    return sample


def gene_non_hallucinatory(sample: dict) -> bool:
    try:
        gene = json.loads(sample["json_output"]).get("GENE")
        return gene is not None and "".join(gene).lower() in sample["sentence"].lower()
    except Exception:
        logger.warning(f"Issue with JSON sample {sample['json_output']}")
        return False


def clean_section(sample: dict) -> dict:
    sample["section"] = str.title(" ".join(sample["section"].split("_")[1:]))
    return sample


def insert_mentions(sample: dict) -> dict:
    # Doing the filtering
    # in here for easier alignment
    # for easier error analysis
    if (
        non_empty_json(sample)
        and gene_non_hallucinatory(sample)
        and attributes_non_empty(sample)
    ):
        components_dict = json.loads(sample["json_output"])
    else:
        components_dict = {}
    mention_components = {"GENE", *ATTRIBUTES}
    for mention_component in mention_components:
        sample[mention_component] = "".join(
            map(str, components_dict.get(mention_component, ["__UNK__"]))
        )
    return sample


def attributes_non_empty(sample: dict, attributes: set[str] = ATTRIBUTES) -> bool:
    return len(attributes & json.loads(sample["json_output"]).keys()) > 0


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def get_files(raw_dir: str) -> Iterable[str]:
    for base_fn in os.listdir(raw_dir):
        yield os.path.join(raw_dir, base_fn)


if __name__ == "__main__":
    main()
