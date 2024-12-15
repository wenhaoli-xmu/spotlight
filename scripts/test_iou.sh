test_scripts=(
    "llama2-7b.json"
    "llama2-7b-upperbound.json"
    "llama2-7b-linearhashing.json"
    "llama2-7b-spotlight.json"
    "llama2-7b-chat.json"
    "llama2-7b-chat-upperbound.json"
    "llama2-7b-chat-linearhashing.json"
    "llama2-7b-chat-spotlight.json"
    "llama3-8b.json"
    "llama3-8b-upperbound.json"
    "llama3-8b-linearhashing.json"
    "llama3-8b-spotlight.json")

for test_script in "${test_scripts[@]}"
do
    echo "Running latency test for ${test_script}..."
    python test_latency/test.py --env_conf "test_latency/${test_script}" --prompt_length $prompt_length

    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done