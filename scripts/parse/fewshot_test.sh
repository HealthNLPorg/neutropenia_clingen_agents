python Repos/cnlp_transformers/src/cnlpt/cnlp_seqgen.py --examples_file batch1/example_1/default_examples.tsv \
       --model_path unsloth/llama-3-8b-Instruct-bnb-4bit \
       --prompt_file batch1/prompting_resources/deepphe_prompt_fewshot.txt \
       --load_in_4bit \
       --max_new_tokens 256 \
       --queries_file batch1/example_1/default_examples_removed_queries.tsv \
       --output_file outputs_test.tsv
