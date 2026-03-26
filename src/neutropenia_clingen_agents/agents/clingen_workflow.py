import argparse
from collections.abc import Collection

import polars as pl
from langgraph.graph.state import END, CompiledStateGraph, StateGraph

from .mention_agent import MentionAgent
from .state_model import ClingenAgentState, Sentence
from .validation_agent import ValidationAgent

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--model_id",
    type=str,
    default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--max_length", type=int, default=8_000)
parser.add_argument("--prompt_file", type=str)
parser.add_argument("--examples_file", type=str)
parser.add_argument("--sample_document", type=str)
parser.add_argument("--sample_answer", type=str)
parser.add_argument("--query_tsv", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--attributes", nargs="+", default={})


def build_agent_workflow(
    model_id: str,
    max_new_tokens: int,
    max_length: int,
    system_prompt: str,
    examples_file: str | None,
    sample_document: str | None,
    sample_answer: str | None,
    attributes: Collection[str] | None,
) -> CompiledStateGraph:
    workflow = StateGraph(Sentence)
    mention_agent_node = MentionAgent(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        system_prompt=system_prompt,
        examples_file=examples_file,
        sample_document=sample_document,
        sample_answer=sample_answer,
    )
    workflow.add_node("mention_agent", mention_agent_node)
    validation_agent_node = (
        ValidationAgent(attributes=attributes)
        if attributes is not None
        else ValidationAgent()
    )
    workflow.add_node("validation_agent", validation_agent_node)
    workflow.set_entry_point("mention_agent")
    workflow.add_edge("mention_agent", "validation_agent")
    workflow.add_edge("validation_agent", END)
    return workflow.compile()


def run_workflow(
    model_id: str,
    max_new_tokens: int,
    max_length: int,
    prompt_file: str,
    query_tsv: str,
    output_dir: str,
    examples_file: str | None,
    sample_document: str | None,
    sample_answer: str | None,
    attributes: Collection[str] | None,
) -> None:
    with open(prompt_file) as f:
        system_prompt = f.read()
    agent_workflow = build_agent_workflow(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        system_prompt=system_prompt,
        examples_file=examples_file,
        sample_document=sample_document,
        sample_answer=sample_answer,
        attributes=attributes,
    )
    df = pl.read_csv(query_tsv, separator="\t")
    sentences = [
        Sentence(
            offsets=(0, len(sentence)),
            sentence_string=sentence,
            raw_output=None,
            mention=None,
        )
        for sentence in df["sentence"]
    ]
    agent_workflow.invoke(ClingenAgentState(sentences=sentences))


def main() -> None:
    args = parser.parse_args()
    run_workflow(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        prompt_file=args.prompt_file,
        query_tsv=args.query_tsv,
        output_dir=args.output_dir,
        examples_file=args.examples_file,
        sample_document=args.sample_document,
        sample_answer=args.sample_answer,
        attributes=args.attributes,
    )


if __name__ == "__main__":
    main()
