from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_huggingface.llms import HuggingFacePipeline
from pydantic import BaseModel, Field

Example = dict[str, str]  # {"input": ..., "output": ...}


class ClinGenMention(BaseModel):
    # Not great attribute naming but we've been using
    # the capital idiom thus far
    GENE: list[str] = Field(
        description="The anchor of any gene mention, the mention does not exist without a gene"
    )
    SYNTAX_N: list[str] = Field(description="Gene variant nucleotide syntax")
    SYNTAX_P: list[str] = Field(description="Gene variant protein syntax")
    VAF: list[str] = Field(description="Variant allele frequency")
    TYPE: list[str] = Field(description="Variant type (pathogentic, unknown, etc.)")
    ZYGOSITY: list[str] = Field(
        description="Whether the variant is heterozygous (VAF > 50%)"
    )


def get_lanchain_hf_pipeline(
    model_id: str,
    model_kwargs: dict,
    pipeline_kwargs: dict,
    task: str = "text-generation",
    device_map: str | None = "auto",
    device: int = -1,
) -> HuggingFacePipeline:
    if device_map is not None:
        return HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task=task,
            device_map=device_map,
            model_kwargs=model_kwargs,
            pipeline_kwargs=pipeline_kwargs,
        )

    return HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task=task,
        device=device,
        model_kwargs=model_kwargs,
        pipeline_kwargs=pipeline_kwargs,
    )


def build_few_shot_prompt_template(
    system_prompt: str, examples: list[Example]
) -> ChatPromptTemplate:
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
    )

    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )
