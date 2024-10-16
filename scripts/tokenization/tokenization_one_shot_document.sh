python Repos/cnlp_transformers/src/cnlpt/cnlp_seqgen.py --model_path unsloth/llama-3-8b-Instruct-bnb-4bit \
       --prompt_file batch1/prompting_resources/tokenization.txt \
       --sample_document batch1/example_1/example_1.txt \
       --sample_answer batch1/example_1/llama3_step1_results.tsv \
       --load_in_4bit \
       --max_new_tokens 2048 \
       --query_files batch1/example_2/example_2 batch1/example_3/example_3 \
       --output_dir ./tokenization_one_shots
