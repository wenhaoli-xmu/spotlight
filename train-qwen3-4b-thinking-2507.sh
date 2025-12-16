MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20000))


torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 4 \
    train.py \
    --num_layers 36 \
    --max_tokens 262144 \
    --model-name-or-path /mnt/petrelfs/share_data/liwenhao/Qwen3-4B-Thinking-2507 \
    --method hash-train \
    --instance_per_cycle 32 \
    --max_prepare_workers 1 \
    --prepare_batch_size_per_gpu 1 \
    --max_top 256 \
    --max_oth 256 \
    --max_que 256 \
    --backward_per_head \
    --max-lr 1e-3 \
    --warmup 0.02 \
    --weight-decay 0.1 \
    --beta1 0.9 \
    --beta2 0.999 \
    --maskout 0.975 \
    --buffer train_buffer_qwen3_4b_thinking_2507 \
    --train-iters 1024 \
    --gradient-clipping 1.0 \
    --gradient-accumulation 1 \
    --train-data "data/github-256k-00000.json" \
    --enable_fsdp
