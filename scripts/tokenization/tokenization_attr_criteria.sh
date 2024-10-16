python Repos/cnlp_transformers/src/cnlpt/cnlp_seqgen.py --model_path unsloth/llama-3-8b-Instruct-bnb-4bit \
       --prompt_file batch1/prompting_resources/tokenization_with_attribute_criteria.txt \
       --load_in_4bit \
       --max_new_tokens 2048 \
       --query_files batch1/example_1/example_1.txt \
       --output_dir ./tokenization_zero_shots
