import logging

from .utils import (
    ClinGenMention,
    build_few_shot_prompt_template,
    get_lanchain_hf_pipeline,
)

Example = dict[str, str]  # {"input": ..., "output": ...}

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class Agent:
    def __init__(
        self,
        system_prompt: str,
        examples: list[Example],
        model_id: str,
        model_kwargs: dict,
        pipeline_kwargs: dict,
    ) -> None:
        self.prompt = build_few_shot_prompt_template(
            system_prompt=system_prompt, examples=examples
        )
        self.model_with_clingen_schema = get_lanchain_hf_pipeline(
            model_id=model_id,
            model_kwargs=model_kwargs,
            pipeline_kwargs=pipeline_kwargs,
        ).with_structured_output(ClinGenMention)
        self.full_chain = self.prompt | self.model_with_clingen_schema

    # Might be more efficient ultimately to do
    # use the batch method with GPU recipe a la
    # https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/#batch-gpu-inference
    # but from the initial standpoint this looks safer given the issues we
    # ran into with calling HF pipelines directly vs using Dataset.map
    # which we ran into in December 2024
    def __call__(self, inputs: list[dict[str, str]]) -> list[ClinGenMention]:
        return self.full_chain.map().invoke(inputs)
