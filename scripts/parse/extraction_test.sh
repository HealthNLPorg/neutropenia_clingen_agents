python ~/Repos/cnlp_transformers/src/cnlpt/cnlp_seqgen.py --model_path unsloth/llama-3-8b-Instruct-bnb-4bit \
       --prompt_file batch1/prompting_resources/syntax_statement_parse.txt \
       --load_in_4bit \
       --max_new_tokens 512 \
       --queries_file batch1/example_1/llama3_step1_results.tsv \
       --output_file syntax_statement_parse.tsv
