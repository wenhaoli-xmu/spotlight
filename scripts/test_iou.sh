test_scripts=(
    "qwen2.5-7b.json")

for test_script in "${test_scripts[@]}"
do
    echo "Running iou test for ${test_script}..."
    python test_iou/test.py --env_conf "test_iou/${test_script}"

    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done