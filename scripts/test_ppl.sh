test_scripts=("llama3-8b-spotlight.json")

for test_script in "${test_scripts[@]}"
do
    echo "Running test for ${test_script}..."
    python test_ppl/test.py --env_conf "test_ppl/${test_script}"
    echo "Finished processing ${test_script}."
    echo "-----------------------------------"
done