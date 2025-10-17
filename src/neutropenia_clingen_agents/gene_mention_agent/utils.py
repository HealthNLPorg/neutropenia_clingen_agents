from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)
from langchain_huggingface.llms import HuggingFacePipeline

Example = dict[str, str]  # {"input": ..., "output": ...}


def get_lanchain_hf_pipeline(
    model_id: str,
    task: str = "text-generation",
    device_map: str | None = "auto",
    device: int = -1,
    **hf_pipeline_kwargs,
) -> HuggingFacePipeline:
    if device_map is not None:
        return HuggingFacePipeline.from_model_id(
            model_id=model_id,
            task=task,
            device_map=device_map,
            pipeline_kwargs=hf_pipeline_kwargs,
        )

    return HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task=task,
        device=device,
        pipeline_kwargs=hf_pipeline_kwargs,
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
