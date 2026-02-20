import logging
from collections.abc import Iterable, Sequence
from functools import partial
from operator import itemgetter
from time import time
from typing import cast

from transformers import pipeline

from ..utils.prompt import get_huggingface_prompt_builder
from .state_model import ClingenAgentState, Document, DocumentSection, Sentence

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class MentionAgent:
    def __init__(
        self,
        model_id: str,
        max_new_tokens: int,
        max_length: int,
        system_prompt: str,
        examples_file: str | None,
        sample_document: str | None,
        sample_answer: str | None,
    ) -> None:
        build_huggingface_prompt = get_huggingface_prompt_builder(
            examples_file, sample_document, sample_answer
        )
        start = time()
        self.model_pipeline = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            max_new_tokens=max_new_tokens,
        )

        end = time()
        logger.info("Loading model took %d seconds", end - start)

        self.local_build_prompt = partial(
            build_huggingface_prompt, system_prompt=system_prompt
        )
        self.local_apply_chat_template = partial(
            self.__apply_chat_template, max_length=max_length
        )

    @staticmethod
    def __parse_raw_output(model_output) -> str:
        return (
            model_output["output"][0]["generated_text"].split("assistant")[-1].strip()
        )

    def __apply_chat_template(
        self, prompted_messages: list[dict[str, str]], max_length: int
    ) -> list[str]:
        if not hasattr(self.model_pipeline.tokenizer, "apply_chat_template"):
            raise NotImplementedError(
                f"{self.model_pipeline.tokenizer} missing apply_chat_template method"
            )
        result = self.model_pipeline.tokenizer.apply_chat_template(
            conversation=prompted_messages,
            tokenize=False,
            add_generation_prompt=False,
            truncate=True,
            max_length=max_length,
        )
        if isinstance(result, list[str]):
            return cast(list[str], result)
        raise ValueError(f"Incorrect result type {type(result)}")

    def __format_to_chat_template(self, inputs: Iterable[str]) -> list[str]:
        prompts = list(map(self.local_build_prompt, inputs))
        return self.local_apply_chat_template(prompts)

    def __predict(self, inputs: list[str]) -> Sequence[str]:
        try:
            return [
                MentionAgent.__parse_raw_output(output)
                for output in self.model_pipeline(inputs)
            ]
        except Exception:
            raise ValueError(f"Ran into issue processing the following sample {inputs}")

    def process_inputs(self, inputs: Iterable[str]) -> Sequence[str]:
        return self.__predict(self.__format_to_chat_template(inputs))

    def __process_section(self, document_section: DocumentSection) -> DocumentSection:
        if len(document_section["sentences"]) == 0:
            section_offsets = document_section["offsets"]
            raise ValueError(f"No sentences in section {section_offsets}")
        if any(
            sentence["raw_output"] is not None or sentence["mention"] is not None
            for sentence in document_section["sentences"]
        ):
            section_offsets = document_section["offsets"]
            raise ValueError(f"One of the sentences in section {section_offsets}")
        raw_outputs = self.process_inputs(
            map(itemgetter("sentence_string"), document_section["sentences"])
        )
        updated_sentences = [
            Sentence(
                offsets=sentence["offsets"],
                sentence_string=sentence["sentence_string"],
                raw_output=raw_output,
                mention=None,
            )
            for raw_output, sentence in zip(raw_outputs, document_section["sentences"])
        ]
        return DocumentSection(
            section_header=document_section["section_header"],
            offsets=document_section["offsets"],
            sentences=updated_sentences,
        )

    def __process_document(self, document: Document) -> Document:
        return Document(
            file_id=document["file_id"],
            sections=[
                self.__process_section(section) for section in document["sections"]
            ],
        )

    def __call__(self, agent_state: ClingenAgentState) -> ClingenAgentState:
        return ClingenAgentState(
            documents=[
                self.__process_document(document)
                for document in agent_state["documents"]
            ]
        )
