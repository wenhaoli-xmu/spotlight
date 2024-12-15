test_scripts=(
    "llama2-7b.json"
    "llama2-7b-spotlight.json"
    "llama2-7b-chat.json"
    "llama2-7b-chat-spotlight.json"
    "llama3-8b.json"
    "llama3-8b-spotlight.json")

for test_script in "${test_scripts[@]}"
do
    echo "Running lmeval for ${test_script}..."
    python test_lmeval/lmeval.py --env_conf "test_lmeval/${test_script}"
    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done