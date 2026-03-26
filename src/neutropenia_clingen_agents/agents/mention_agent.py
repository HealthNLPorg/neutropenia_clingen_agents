import logging
from collections.abc import Iterable, Sequence
from functools import partial
from time import time
from typing import cast

from transformers import pipeline

from ..utils.prompt import get_huggingface_prompt_builder
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
        max_length: int,
        system_prompt: str,
        examples_file: str | None,
        sample_document: str | None,
        sample_answer: str | None,
    ) -> None:
        build_huggingface_prompt = get_huggingface_prompt_builder(
            examples_file=examples_file,
            sample_document=sample_document,
            sample_answer=sample_answer,
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

        self.local_build_prompt = partial(build_huggingface_prompt, system_prompt)
        self.local_apply_chat_template = partial(
            self.__apply_chat_template, max_length=max_length
        )

    @staticmethod
    def __parse_raw_output(model_output) -> str:
        return model_output[0]["generated_text"].split("assistant")[-1].strip()

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
        if isinstance(result, list) and all(
            map(lambda elem: isinstance(elem, str), result)
        ):
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
