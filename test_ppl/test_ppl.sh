#!/bin/bash

models=(
    "/pfs/rl-train/wenhaoli/gdrive/model/Qwen3-4B"
)

methods=(
    "origin"
    "hash-eval"
)

for model in "${models[@]}"
do
    for method in "${methods[@]}"
    do
        echo "Running PPL test for $(basename "$model")-${method}..."
        python test_ppl/test.py \
            --model-name-or-path "$model" \
            --method "$method"

        echo "Finished processing ${model}-${method}."
        echo "-----------------------------------"
    done
done