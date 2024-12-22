import argparse
import json
import logging
import os
import pathlib
import re
from itertools import chain
from time import time
from typing import Callable, Dict, Iterable, List, Tuple, cast

import pandas as pd
from datasets import Dataset, load_dataset
from transformers import pipeline

# from transformers.pipelines.pt_utils import KeyDataset

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
    "--sample_answer",
    type=str,
)
parser.add_argument("--prompt_file", type=str)
parser.add_argument(
    "--model_path",
    type=str,
    default="/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
)

parser.add_argument(
    "--attn_implementation",
    type=str,
    default="spda",
    choices=["spda", "flash_attention_2"],
)

parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--load_in_8bit", action="store_true")
parser.add_argument("--fancy_output", action="store_true")
parser.add_argument("--model_name", choices=["llama2", "llama3", "mixtral", "qwen2"])


parser.add_argument(
    "--max_new_tokens",
    type=int,
    help="1 for classification, on the order of 128 for BIO, on the order of 1024 for free text analysis and explanation",
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

# {role: {system|user|assistant}, content: ...}
Message = Dict[str, str]

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


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
    system_prompt = get_system_prompt(args.prompt_file)
    logger.info("Building dataset")
    query_dataset = load_dataset(
        "csv",
        sep="\t",
        data_files=os.listdir(args.query_dir) if args.query_dir else args.query_files,
    )
    query_dataset = query_dataset["train"]
    logger.info(f"OVER HERE {query_dataset}")

    def few_shot_with_examples(
        examples: Iterable[Tuple[str, str]]
    ) -> Callable[[str, str], List[Message]]:
        def _few_shot_prompt(s, q):
            return few_shot_prompt(system_prompt=s, query=q, examples=examples)

        return _few_shot_prompt

    if args.examples_file is not None:
        examples = get_examples(args.examples_file)
        if len(examples) > 0:
            get_prompt = few_shot_with_examples(examples=examples)

        else:
            ValueError("Empty examples file")

            get_prompt = empty_prompt
    elif args.sample_document is not None and args.sample_answer is not None:
        example = get_document_level_example(args.sample_document, args.sample_answer)
        if all(len(ex) > 0 for ex in example):
            get_prompt = few_shot_with_examples(examples=(example,))
        else:
            ValueError("Empty sample document and/or empty sample answer")

            get_prompt = empty_prompt
    else:
        get_prompt = zero_shot_prompt
    start = time()
    seqgen_pipe = pipeline(
        "text-generation",
        model=final_path,
        # use_auth_token=True,
        device_map="auto",
        model_kwargs={"load_in_4bit": True},
        max_new_tokens=args.max_new_tokens,
        batch_size=32,
    )
    end = time()
    logger.info(f"Loading model took {end-start} seconds")
    # current_time = datetime.datetime.now(pytz.timezone("America/New_York"))
    out_dir = args.output_dir
    out_fn_stem = pathlib.Path(
        args.query_dir
        if args.query_dir
        else "_".join(basename_no_ext(fn) for fn in args.query_files)
    ).stem
    # out_fn = f"{out_fn_stem}.txt"
    excel_out_fn = f"{out_fn_stem}.xlsx"
    excel_out_path = os.path.join(out_dir, excel_out_fn)
    tsv_out_fn = f"{out_fn_stem}.tsv"
    tsv_out_path = os.path.join(out_dir, tsv_out_fn)

    def format_chat(sample: dict) -> dict:
        return {
            "text": seqgen_pipe.tokenizer.apply_chat_template(
                get_prompt(system_prompt, sample["sentence"]),
                tokenize=False,
                add_generation_prompt=False,
            )
        }

    def predict(batch):
        batch["output"] = seqgen_pipe(batch["text"])
        return batch

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    query_dataset = query_dataset.map(format_chat)
    logger.info(f"Processed dataset for {query_dataset}")
    logger.info(f"Starting model inference on {query_dataset}")
    query_dataset = query_dataset.map(
        predict,
        batched=True,
        batch_size=32,
        # with_rank=True,
        # num_proc=torch.cuda.device_count(),
    )
    logger.info(f"{query_dataset} inference finished")
    query_dataset = query_dataset.map(parse_output)
    query_dataset = query_dataset.filter(non_empty_json)
    query_dataset = query_dataset.filter(gene_non_hallucinatory)
    query_dataset = query_dataset.filter(attributes_non_empty)
    query_dataframe = query_dataset.to_pandas()
    query_dataframe.to_excel(excel_out_path)
    query_dataframe.to_csv(tsv_out_path, sep="\t")


def try_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def non_empty_json(sample: dict) -> bool:
    return len(sample["json_output"]) > 0


def parse_output(sample: dict) -> dict:
    model_answer = sample["output"][0]["generated_text"].split("assistant")[-1].strip()
    sample["json_output"] = try_json(model_answer)
    return sample


def gene_non_hallucinatory(sample: dict) -> bool:
    gene = sample["json_output"].get("GENE")
    try:
        return gene is not None and gene.lower() in sample["sentence"].lower()
    except Exception:
        print(gene)


def attributes_non_empty(sample: dict) -> bool:
    return len({"STATEMENT", "SYNTAX_N", "SYNTAX_P"} & sample["json_output"].keys()) > 0


def empty_prompt(system_prompt: str, query: str) -> List[Message]:
    return []


def structure_response(index: int, query: str, answer: str) -> str:
    return f"Query {index}:\n{query}\nAnswer:\n{answer}\n\n"


def basename_no_ext(fn: str) -> str:
    return pathlib.Path(fn).stem.strip()


def get_system_prompt(prompt_file_path: str) -> str:
    with open(prompt_file_path, mode="rt", encoding="utf-8") as f:
        raw_prompt = f.read()
        # cleaned_prompt = " ".join(raw_prompt.strip().split())
        # return cleaned_prompt
        return raw_prompt


def get_query_dataset(queries_file_path: str) -> Dataset:
    # NB, this will retrieve the extension with the "." at the front
    # e.g. ".txt" rather than "txt"
    suffix = pathlib.Path(queries_file_path).suffix.lower()
    match suffix.strip():
        case ".tsv":
            full_dataframe = pd.read_csv(queries_file_path, sep="\t")
            raw_queries = cast(
                Iterable[str],
                (
                    full_dataframe["query"]
                    if "query" in full_dataframe.columns
                    else full_dataframe["sentence"]
                ),
            )

            def with_whitespace() -> Iterable[Dict[str, str]]:
                for query in raw_queries:
                    yield {"text": reinsert_whitespace(query)}

            queries = Dataset.from_generator(with_whitespace)
        case ".txt" | "":
            with open(queries_file_path, mode="rt") as qf:
                query = qf.read()
            queries = Dataset.from_list([{"text": query}])
        case _:
            ValueError(f"Presently unsupported query format {suffix}")
            queries = Dataset.from_list([])
    return queries


def get_examples(examples_file_path: str) -> List[Tuple[str, str]]:
    suffix = pathlib.Path(examples_file_path).suffix.lower()
    match suffix.strip():
        case ".tsv":
            full_dataframe = pd.read_csv(examples_file_path, sep="\t")
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


def parse_input_output(examples_file_path: str) -> List[Tuple[str, str]]:
    def parse_example(raw_example: str) -> Tuple[str, str]:
        result = tuple(
            elem.strip()
            for elem in re.split("input:|output:", raw_example)
            if len(elem.strip()) > 0
        )
        assert len(result) == 2
        return result

    with open(examples_file_path, mode="rt", encoding="utf-8") as ef:
        raw_str = ef.read()
        return [
            parse_example(example.strip())
            for example in raw_str.split("\n\n")
            if len(example.split()) > 0
        ]


def get_document_level_example(
    sample_document_path: str, sample_answer_path: str
) -> Tuple[str, str]:
    with open(sample_document_path, mode="rt", encoding="utf-8") as sample_document:
        # not normalizing newlines since those might be useful
        query = sample_document.read()
    sample_answer_dataframe = pd.read_csv(sample_answer_path, sep="\t")
    # specific to earlier use-case etc but for now
    answer = "\n".join(cast(Iterable[str], sample_answer_dataframe["query"]))
    return (query, answer)


def zero_shot_prompt(system_prompt: str, query: str) -> List[Message]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    return messages


# retain newline information via special markers
# while removing them for storage
# ( so you can load them later via pandas without parsing errors )
def clean_whitespace(sample: str) -> str:
    return (
        sample.replace("\n", "<cn>")
        .replace("\t", "<ct>")
        .replace("\f", "<cf>")
        .replace("\r", "<cr>")
    )


def reinsert_whitespace(sample: str) -> str:
    return (
        sample.replace("<cn>", "\n")
        .replace("<ct>", "\t")
        .replace("<cf>", "\f")
        .replace("<cr>", "\r")
    )


def few_shot_prompt(
    system_prompt: str, query: str, examples: Iterable[Tuple[str, str]]
) -> List[Message]:
    def message_pair(ex_query: str, ex_answer: str) -> Tuple[Message, ...]:
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


def get_files(raw_dir: str) -> Iterable[str]:
    for base_fn in os.listdir(raw_dir):
        yield os.path.join(raw_dir, base_fn)


if __name__ == "__main__":
    main()
