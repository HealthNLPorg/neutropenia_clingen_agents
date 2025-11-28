import argparse
import json
import logging
import os
import pathlib

from datasets import Dataset, load_dataset

from .utils.filesystem import make_directory

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--processed_tsv",
    type=str,
)

parser.add_argument("--output_dir", type=str)
logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

ATTRIBUTES = {"VAF", "SYNTAX_N", "SYNTAX_P", "TYPE"}


def post_process_from_tsv(
    processed_tsv: str,
    output_dir: str,
) -> None:
    processed_dataset = load_dataset(
        "csv",
        sep="\t",
        data_files=(processed_tsv,),
    )
    make_directory(output_dir)
    query_tsv_stem = pathlib.Path(processed_tsv).stem
    post_processed_tsv_query_tsv = f"post_processed_{query_tsv_stem}.tsv"
    post_processed_tsv_out_path = os.path.join(output_dir, post_processed_tsv_query_tsv)
    post_process_dataset(processed_dataset, post_processed_tsv_out_path)


def post_process_dataset(processed_dataset: Dataset, tsv_out_path: str) -> None:
    processed_dataset = (
        processed_dataset.map(parse_output)
        .map(insert_mentions)
        .map(clean_section)
        .remove_columns(["text", "output", "json_output", "raw_output"])
    )
    processed_dataframe = processed_dataset.to_polars()
    renamed_column_mapping = {col: col.title() for col in processed_dataframe.columns}
    processed_dataframe = processed_dataframe.rename(mapping=renamed_column_mapping)
    processed_dataframe = processed_dataframe[
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
    processed_dataframe.write_csv(tsv_out_path, separator="\t")


def try_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        return {}


def non_empty_json(sample: dict) -> bool:
    return len(sample["json_output"]) > 0


def parse_output(sample: dict) -> dict:
    sample["json_output"] = json.dumps(try_json(sample["raw_output"]))
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


def main() -> None:
    args = parser.parse_args()
    post_process_from_tsv(
        args.processed_tsv,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
