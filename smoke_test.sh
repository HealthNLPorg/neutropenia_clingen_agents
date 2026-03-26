uv run python -m neutropenia_clingen_agents.agents.clingen_workflow \
    --model_id unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit \
    --max_new_tokens 512 \
    --query_tsv ./smoke_test.tsv \
    --output_dir ./test \
# --system_prompt_file ./resources/prompts/vtype_fixed.txt \
# --examples_file ./resources/examples/ex12_and_invitae.tsv \
