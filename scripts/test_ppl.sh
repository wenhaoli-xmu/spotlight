test_scripts=(
    "llama3-8b-upperbound.json"
    "llama2-7b-upperbound.json"
    "llama2-7b-chat-upperbound.json"
    "qwen2.5-7b-upperbound.json"
    "qwen2.5-7b-spotlight.json")

for test_script in "${test_scripts[@]}"
do
    echo "Running test for ${test_script}..."
    python test_ppl/test.py --env_conf "test_ppl/${test_script}"
    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done