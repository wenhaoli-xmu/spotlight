MASTER_ADDR=localhost
MASTER_PORT=$((RANDOM % 101 + 20000))

RUN_NAME=test
BUFFER=train_buffer_qwen3_4b_buffer

torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    --nnodes 1 \
    --nproc_per_node 8 \
    train.py \
    --num_layers 32 \
    --max_tokens 40960 \
    --model-name-or-path /pfs/rl-train/wenhaoli/gdrive/model/Qwen3-4B \
    --method hash-train \
    --instance_per_cycle 64 \
    --max_prepare_workers 8 \
    --prepare_batch_size_per_gpu 1 \
    --maskout 0.975 \
    --max_que 256 \
    --max_top 256 \
    --max_oth 256 \
    --max-lr 1e-3 \
    --warmup 0 \
    --weight-decay 0.01 \
    --beta1 0.9 \
    --beta2 0.999 \
    --buffer $BUFFER \
    --train-iters 6144 \
    --gradient-clipping 1.0 \
    --gradient-accumulation 1 \
    --run-name $RUN_NAME
