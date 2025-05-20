test_scripts=(
    "llama2-7b-spotlight.json"
    "llama2-7b-chat-spotlight.json"
    "llama3-8b-spotlight.json"
    "qwen2.5-7b-spotlight"
    "llama2-7b-upperbound.json"
    "llama2-7b-chat-upperbound.json"
    "llama3-8b-upperbound.json"
    "qwen2.5-7b-upperbound")

for test_script in "${test_scripts[@]}"
do
    echo "Running iou test for ${test_script}..."
    python test_iou/test.py --env_conf "test_iou/${test_script}"

    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done