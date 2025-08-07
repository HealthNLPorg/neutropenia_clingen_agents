import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

# These are useful
# https://github.com/anand-subu/blog_resources/blob/main/introduction_to_prompting/helpers.py
# https://towardsdatascience.com/building-a-biomedical-entity-linker-with-llms-d385cb85c15a
# https://towardsdatascience.com/an-introduction-to-prompting-for-llms-61d36aec2048
# Few shots in
#  r/neutropenia_registry/samples_from_karyn/few_shot_genes.txt
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--examples_file",
    type=str,
)
parser.add_argument("--prompt_file", type=str)
parser.add_argument(
    "--model_path",
    type=str,
    default="/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
)
parser.add_argument("--model_name", choices=["llama2", "llama3", "mixtral", "qwen2"])


parser.add_argument("--input_file", type=str, help="TSVs if you please")
parser.add_argument("--output_file", type=str)

name2path = {
    "llama2": "/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
    "llama3": "/lab-share/CHIP-Savova-e2/Public/resources/Meta-Llama-3-8B-Instruct/",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "qwen2": "Qwen/Qwen2-1.5B-Instruct",
}


# Will need to change this for new formatting
def create_query(item):
    """
    Creates the input for the model using the question and the multiple choice options.

    Args:
        item (dict): A dictionary containing the question and options.
            Expected keys are "question" and "options", where "options" is another
            dictionary with keys "A", "B", "C", and "D".

    Returns:
        str: A formatted query combining the question and options, ready for use.
    """
    query = (
        item["question"]
        + "\nOptions:\n"
        + "A. "
        + item["options"]["A"]
        + "\n"
        + "B. "
        + item["options"]["B"]
        + "\n"
        + "C. "
        + item["options"]["C"]
        + "\n"
        + "D. "
        + item["options"]["D"]
    )
    return query


def build_zero_shot_prompt(system_prompt, question):
    """
    Builds the zero-shot prompt.

    Args:
        system_prompt (str): Task Instruction for the LLM
        content (dict): The content for which to create a query, formatted as
            required by `create_query`.

    Returns:
        list of dict: A list of messages, including a system message defining
            the task and a user message with the input question.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": create_query(question)},
    ]
    return messages


def build_few_shot_prompt(system_prompt, content, few_shot_examples):
    """
    Builds the few-shot prompt using provided examples.

    Args:
        system_prompt (str): Task description for the LLM
        content (dict): The content for which to create a query, similar to the
            structure required by `create_query`.
        few_shot_examples (list of dict): Examples to simulate a hypothetical
            conversation. Each dict must have "options" and an "answer".

    Returns:
        list of dict: A list of messages, simulating a conversation with
            few-shot examples, followed by the current user query.
    """
    messages = [{"role": "system", "content": system_prompt}]
    for item in few_shot_examples:
        ans_options = item["options"]
        correct_ans_option = ""
        for key, value in ans_options.items():
            if value == item["answer"]:
                correct_ans_option = key
                break
        messages.append({"role": "user", "content": create_query(item)})
        messages.append(
            {
                "role": "assistant",
                "content": "The answer is " + correct_ans_option + ".",
            }
        )
    messages.append({"role": "user", "content": create_query(content)})
    return messages


def entry() -> None:
    args = parser.parse_args()
    final_path = ""
    if args.model_name is not None:
        final_path = name2path[args.model_name]
    else:
        final_path = args.model_path
    tokenizer = AutoTokenizer.from_pretrained(final_path, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        final_path, load_in_4bit=True, use_auth_token=True
    )


def main() -> None:
    entry()


if __name__ == "__main__":
    main()
