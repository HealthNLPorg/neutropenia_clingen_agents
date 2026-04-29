import argparse
from collections.abc import Collection

import polars as pl
from langgraph.graph.state import END, CompiledStateGraph, StateGraph

from .mention_agent import MentionAgent
from .state_model import Sentence

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
parser.add_argument("--anchor", type=str, default=None)
parser.add_argument("--attributes", nargs="+", default=None)


def quickstart() -> CompiledStateGraph:
    workflow = StateGraph(Sentence)
    from .resources.examples.default import examples
    from .resources.prompts.vtype_fixed import prompt

    mention_agent_node = MentionAgent(
        model_id="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_new_tokens=512,
        system_prompt=prompt,
        examples=examples,
    )
    workflow.add_node("mention_agent", mention_agent_node.process_sentence)
    workflow.set_entry_point("mention_agent")
    workflow.add_edge("mention_agent", END)
    return workflow.compile()


def build_agent_workflow(
    model_id: str,
    max_new_tokens: int,
    system_prompt: str,
    examples_file: str | None,
    sample_document: str | None,
    sample_answer: str | None,
    anchor: str | None,
    attributes: Collection[str] | None,
) -> CompiledStateGraph:
    workflow = StateGraph(Sentence)
    mention_agent_node = MentionAgent(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
        examples_file=examples_file,
        examples=None,
        sample_document=sample_document,
        sample_answer=sample_answer,
    )
    workflow.add_node("mention_agent", mention_agent_node.process_sentence)
    workflow.set_entry_point("mention_agent")
    workflow.add_edge("mention_agent", END)
    return workflow.compile()


def run_workflow(
    model_id: str,
    max_new_tokens: int,
    prompt_file: str,
    query_tsv: str,
    examples_file: str | None,
    sample_document: str | None,
    sample_answer: str | None,
    anchor: str | None,
    attributes: Collection[str] | None,
) -> None:
    with open(prompt_file) as f:
        system_prompt = f.read()
    agent_workflow = build_agent_workflow(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
        examples_file=examples_file,
        sample_document=sample_document,
        sample_answer=sample_answer,
        anchor=anchor,
        attributes=attributes,
    )
    df = pl.read_csv(query_tsv, separator="\t")
    for sentence in df["sentence"]:
        print(
            agent_workflow.invoke(
                Sentence(
                    offsets=(0, len(sentence)),
                    sentence_string=sentence,
                    raw_output=None,
                    mention=None,
                )
            )
        )


def main() -> None:
    args = parser.parse_args()
    run_workflow(
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        prompt_file=args.prompt_file,
        query_tsv=args.query_tsv,
        examples_file=args.examples_file,
        sample_document=args.sample_document,
        sample_answer=args.sample_answer,
        anchor=args.anchor,
        attributes=args.attributes,
    )


if __name__ == "__main__":
    main()
