python Repos/cnlp_transformers/src/cnlpt/cnlp_seqgen.py --model_path unsloth/llama-3-8b-Instruct-bnb-4bit \
       --examples batch1/example_1/examples_for_step_2.tsv \
       --prompt_file batch1/prompting_resources/syntax_statement_parse.txt \
       --load_in_4bit \
       --max_new_tokens 512 \
       --queries_file batch1/example_1/step_1_few_show_examples_removed.tsv \
       --output_file syntax_statement_parse_few_shot.tsv
