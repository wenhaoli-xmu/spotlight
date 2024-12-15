test_scripts=(
    "llama2-7b-maskout98.json"
    "llama2-7b-maskout90.json"
    "llama2-7b-chat-maskout98.json"
    "llama2-7b-chat-maskout90.json")

for test_script in "${test_scripts[@]}"
do
    echo "Processing ${test_script}..."
    python mmlu/evaluate.py --env_conf "mmlu/${test_script}"
    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done