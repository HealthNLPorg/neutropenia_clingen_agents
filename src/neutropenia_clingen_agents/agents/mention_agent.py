import logging
from collections.abc import Iterable, Sequence
from functools import partial
from time import time
from typing import cast

from transformers import pipeline

from ..utils.prompt import (
    few_shot_prompt,
    get_huggingface_prompt_builder,
    zero_shot_prompt,
)
from .state_model import Sentence

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
        system_prompt: str,
        examples_file: str | None = None,
        examples: Sequence[tuple[str, str]] | None = None,
        sample_document: str | None = None,
        sample_answer: str | None = None,
    ) -> None:
        if examples_file is not None and examples is None:
            self.build_prompt = get_huggingface_prompt_builder(
                examples_file=examples_file,
                sample_document=sample_document,
                sample_answer=sample_answer,
            )
        elif examples_file is None and examples is not None:
            self.build_prompt = partial(few_shot_prompt, examples=examples)
        else:
            logger.info("No examples or examples file provided, using zero shot")
            self.build_prompt = zero_shot_prompt
        self.system_prompt = system_prompt
        start = time()
        self.model_pipeline = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
            max_new_tokens=max_new_tokens,
        )

        end = time()
        logger.info("Loading model took %d seconds", end - start)

    @staticmethod
    def __parse_raw_output(model_output) -> str:
        return model_output[0]["generated_text"].split("assistant")[-1].strip()

    def __apply_chat_template(
        self, prompted_messages: list[list[dict[str, str]]]
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
        )
        if isinstance(result, list) and all(
            map(lambda elem: isinstance(elem, str), result)
        ):
            return cast(list[str], result)
        raise ValueError(f"Incorrect result type {type(result)}")

    def __format_to_chat_template(self, queries: Iterable[str]) -> list[str]:
        prompts = [
            [
                {str(k): str(v) for k, v in message_mapping.items()}
                for message_mapping in self.build_prompt(self.system_prompt, query)
            ]
            for query in queries
        ]
        return self.__apply_chat_template(prompts)

    def __predict(self, inputs: list[str]) -> Sequence[str]:
        try:
            return [
                MentionAgent.__parse_raw_output(output)
                for output in self.model_pipeline(inputs)
            ]
        except Exception as e:
            raise ValueError(f"Ran into {e} processing the following sample {inputs}")

    def process_inputs(self, inputs: Iterable[str]) -> Sequence[str]:
        return self.__predict(self.__format_to_chat_template(inputs))

    def process_sentence(self, sentence: Sentence) -> Sentence:
        return Sentence(
            offsets=sentence.offsets,
            sentence_string=sentence.sentence_string,
            raw_output=self.process_inputs([sentence.sentence_string])[0],
            mention=None,
        )
