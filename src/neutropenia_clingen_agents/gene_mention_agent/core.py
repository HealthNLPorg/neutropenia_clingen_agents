import logging

from langchain_core.output_parsers import PydanticOutputParser

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
        self.parser = PydanticOutputParser(pydantic_object=ClinGenMention)
        self.prompt = build_few_shot_prompt_template(
            system_prompt=system_prompt, examples=examples, parser=self.parser
        )
        self.model_with_clingen_schema = get_lanchain_hf_pipeline(
            model_id=model_id,
            model_kwargs=model_kwargs,
            pipeline_kwargs=pipeline_kwargs,
        )
        logger.info(f"What the prompt looks like - {self.prompt}")
        # self.full_chain = self.prompt | self.model_with_clingen_schema  # | self.parser

    # Might be more efficient ultimately to do
    # use the batch method with GPU recipe a la
    # https://python.langchain.com/docs/integrations/llms/huggingface_pipelines/#batch-gpu-inference
    # but from the initial standpoint this looks safer given the issues we
    # ran into with calling HF pipelines directly vs using Dataset.map
    # which we ran into in December 2024
    def __call__(self, inputs: list[dict[str, str]]) -> list[ClinGenMention]:
        logger.info(f"__call__ reached for {inputs}")
        # results = self.full_chain.map().invoke(inputs)
        results = []
        for instance in inputs:
            logger.info(f"Running for instance {instance}")
            try:
                instance = self.prompt.invoke(instance)
                result = self.model_with_clingen_schema.invoke(
                    instance,
                    {
                        "pipeline_kwargs": {
                            "tokenizer_encode_kwargs": {
                                "truncate": True,
                                "add_generation_prompt": False,
                                "tokenize": False,
                            }
                        }
                    },
                )
                logger.info(f"Result {result}")
            except Exception as e:
                logger.error(e)
                result = {}
            results.append(result)
            # try:
            #     prompt_result = self.prompt.invoke(instance)
            #     logger.info(f"Prompt result: {prompt_result}")
            # except Exception as e:
            #     logger.error(e)
            #     exit(1)
            # model_result = self.model_with_clingen_schema.invoke(prompt_result)
            # logger.info(f"Model result: {model_result}")
            # try:
            #     parser_result = self.parser.invoke(model_result)
            #     logger.info(f"Parser result: {parser_result}")
            # except Exception as e:
            #     logger.error(e)
            # finally:
            #     parser_result = {}
            # results.append(parser_result)
        logger.info(f"Results - {results}")
        return results
