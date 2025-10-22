import logging

from langchain_core.runnables import Runnable

from .utils import build_few_shot_prompt_template, get_lanchain_hf_pipeline

Example = dict[str, str]  # {"input": ..., "output": ...}

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_llm_chain(
    system_prompt: str,
    examples: list[Example],
    model_id: str,
    model_kwargs: dict,
    pipeline_kwargs: dict,
) -> Runnable:
    few_shot_prompt_template = build_few_shot_prompt_template(
        system_prompt=system_prompt, examples=examples
    )
    langchain_hf_pipeline = get_lanchain_hf_pipeline(
        model_id=model_id, model_kwargs=model_kwargs, pipeline_kwargs=pipeline_kwargs
    )
    return few_shot_prompt_template | langchain_hf_pipeline


class Agent:
    def __init__(
        self,
        system_prompt: str,
        examples: list[Example],
        model_id: str,
        model_kwargs: dict,
        pipeline_kwargs: dict,
    ) -> None:
        self.chain = get_llm_chain(
            system_prompt=system_prompt,
            examples=examples,
            model_id=model_id,
            model_kwargs=model_kwargs,
            pipeline_kwargs=pipeline_kwargs,
        )

    # Might be more efficient ultimately to do
    # use the batch method with GPU recipe a la
    # https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/#batch-gpu-inference
    # but from the initial standpoint this looks safer given the issues we
    # ran into with calling HF pipelines directly vs using Dataset.map
    # which we ran into in December 2024
    def __call__(self, inputs: list[dict[str, str]]) -> list[str]:
        return self.chain.map().invoke(inputs)
