input_dir=../crc_llama2_testcase/
output_dir=.
cancer_type=crc
# input_filename=${cancer_type}_dev_for_llm.json
input_filename=sample.json

python ./fewshot_llama2.py --input_dir $input_dir \
       --input_filename $input_filename \
       --cancer_type $cancer_type \
       --output_dir $output_dir \
       --few_shot_example_path timenorm_examples.txt > \
       llama2_test_case.txt
