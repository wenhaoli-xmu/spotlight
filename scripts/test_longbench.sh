#!/bin/bash

models=(
    "/mnt/petrelfs/share_data/liwenhao/Qwen3-4B-Thinking-2507"
)

methods=(
    "origin"
)


MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((RANDOM % 101 + 20000))

for model in "${models[@]}"
do
    for method in "${methods[@]}"
    do
        model_name=$(basename "$model")
        save_dir="${model_name}-${method}"
        
        mkdir -p "pred/$save_dir"

        echo "Running prediction for ${save_dir}..."

        # torchrun \
        #     --rdzv-backend=c10d \
        #     --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
        #     --nnodes 1 \
        #     --nproc_per_node 8 \
        #     test_longbench/pred_multigpu.py \
        #     --model-name-or-path "$model" \
        #     --method "$method"

        # python test_longbench/pred.py \
        #     --model-name-or-path "$model" \
        #     --method "$method"

        echo "Evaluating model for ${save_dir}..."
        python LongBench/eval.py --model "${save_dir}"

        echo "Displaying results for ${save_dir}..."
        python test_longbench/sort.py "${save_dir}"

        echo "Finished processing ${save_dir}."
        echo "-----------------------------------"
    done
done