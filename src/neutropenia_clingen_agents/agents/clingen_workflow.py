from collections.abc import Collection

from langgraph.graph.state import END, CompiledStateGraph, StateGraph

from .mention_agent import MentionAgent
from .state_model import ClingenAgentState
from .validation_agent import ValidationAgent


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
    workflow = StateGraph(state_schema=ClingenAgentState)
    mention_agent_node = MentionAgent(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        max_length=max_length,
        system_prompt=system_prompt,
        examples_file=examples_file,
        sample_document=sample_document,
        sample_answer=sample_answer,
    )
    workflow.add_node(node="mention_agent", action=mention_agent_node)
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
