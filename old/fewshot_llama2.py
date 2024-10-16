import argparse
import json
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_dir", type=str)
parser.add_argument("--input_filename", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--output_filename", type=str)
parser.add_argument("--cancer_type", type=str)
parser.add_argument(
    "--few_shot_example_path",
    type=str,
    default="fewshot_sentbysent_gold_examples.txt",
)
parser.add_argument(
    "--model_path",
    type=str,
    default="/lab-share/CHIP-Savova-e2/Public/resources/llama-2/Llama-2-70b-chat-hf",
)


def read_in_fewshot_examples(input_file):
    with open(input_file, "r") as fr:
        four_shot_examples = fr.read()
    return four_shot_examples


def read_input_data(input_file):
    with open(input_file, "r") as fr:
        input_data = json.load(fr)

    timeline_model_input = []
    normalization_input = []
    info = []
    for doc in input_data:
        for sent in doc["sentences"]:
            timeline_model_input.append(sent)
            normalization_input.append(doc["doc_time"])
            info.append((doc["doc_id"], doc["note_id"]))
    return timeline_model_input, normalization_input, info


def get_prompt(system_prompt, timeline_query, prompt_for_test_case, examples=[]):
    timeline_query = timeline_query.replace("<EOL> ", "")
    prompt = system_prompt + "\n"
    if isinstance(examples, list):
        for e in examples:
            prompt += e
            prompt += "\n"
    else:
        isinstance(examples, str)
        prompt += examples
        prompt += "\n"
    prompt += prompt_for_test_case
    prompt += "\n"
    prompt += "PATIENT EHR: \n" + timeline_query + "\nTIMELINES: \n" + "[/INST]"
    return prompt


def get_llama_output(prompt, pipeline, tokenizer):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=2048,
    )
    generated_text = sequences[0]["generated_text"]
    response = generated_text[len(prompt) :]  # Remove the prompt from the output
    print("LLAMA2:", response.strip())
    return {"PROMPT": prompt, "MODEL_OUTPUT": response.strip()}


def get_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_4bit=True, use_auth_token=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    llama_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return llama_pipeline, tokenizer


def prompt_loop(
    cancer_type: str,
    few_shot_example_path: str,
    input_filename: str,
    model_path: str,
    input_dir: str,
    output_dir: str,
) -> None:

    SYSTEM_PROMPT = (
        "<s>[INST] <<SYS>>\nYou are a helpful assistant for oncologists. "
        "You will read the given PATIENT EHR and summarize the patient's chemotherapy treatment TIMELINES. "
        "Please only output TIMELINES in the requested format. Please do not include "
        "any other text or reasoning, do not include timelines for any other treatments besides chemotherapy. "
        "Please do not use any labels other than the ones given in the examples, "
        "i.e. BEGINS-ON, ENDS-ON, CONTAINS. Here are some examples. <</SYS>>"
    )

    prompt_for_test_case = SYSTEM_PROMPT.split("\n")[-1].replace("<</SYS>>", "")
    output_name = os.path.join(
        output_dir,
        cancer_type + "_dev_llama2_70b_ctakesChemo_bysent_step1.json",
    )

    # few shot examples pass to the model
    EXAMPLES = read_in_fewshot_examples(few_shot_example_path)
    # Read in your data
    timeline_model_input, normalization_input, info = read_input_data(
        os.path.join(input_dir, input_filename)
    )
    all_output = []
    llama_pipeline, tokenizer = get_model(model_path)
    for idx, instance in enumerate(timeline_model_input):
        norm_instance = normalization_input[idx]
        prompt = get_prompt(SYSTEM_PROMPT, instance, prompt_for_test_case, EXAMPLES)
        print("\n", "start...")
        print("PROMPT: ", prompt)
        print()
        # Get output from llama
        model_output = get_llama_output(prompt, llama_pipeline, tokenizer)
        # Write llama output somewhere
        all_output.append(
            {
                "model_output": model_output["MODEL_OUTPUT"],
                "doc_time": norm_instance,
                "info": info[idx],
            }
        )
    with open(output_name, "w") as fw:
        json.dump(all_output, fw)


def main():
    args = parser.parse_args()
    prompt_loop(
        args.cancer_type,
        args.few_shot_example_path,
        args.input_filename,
        args.model_path,
        args.input_dir,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
