#!/bin/bash

# Define parameters
chat_template=default
s_len=256
e_len=8192
step=256

# Define configurations
configs=("llama3-8b.json" "llama3-8b-spotlight.json")

# Loop through each configuration
for config in "${configs[@]}"; do
    echo "Running with config: $config"
    
    # Run the needle in haystack
    python test_needle/run_needle_in_haystack.py \
        --env_conf test_needle/$config \
        --chat_template $chat_template \
        --s_len $s_len \
        --e_len $e_len \
        --step $step

    # Visualize the result
    python test_needle/viz.py \
        --env_conf test_needle/$config
done