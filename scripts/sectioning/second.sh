cd ~
python Repos/cnlp_transformers/src/cnlpt/cnlp_seqgen.py --model_path unsloth/llama-3-8b-Instruct-bnb-4bit \
       --prompt_file ~/Repos/prompting/section_parsing/second.txt \
       --load_in_4bit \
       --max_new_tokens 2048 \
       --query_dir ~/batch1/all_text_files/ \
       --output_dir ~/neutropenia_prompting_experiment_results/section_parsing_experiments/second_iteration/
