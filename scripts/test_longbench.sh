test_scripts=("llama2-7b-chat-magicpig.json")

model_max_length=4096

for test_script in "${test_scripts[@]}"
do
    rm -r pred/$test_script
    mkdir pred/$test_script

    echo "Running prediction for ${test_script}..."
    python test_longbench/pred.py --env_conf "test_longbench/${test_script}" --chat_template llama-2 --magicpig --model_max_length $model_max_length

    echo "Evaluating model for ${test_script}..."
    python LongBench/eval.py --model "${test_script}"

    echo "Displaying results for ${test_script}..."
    python test_longbench/sort.py $test_script

    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done