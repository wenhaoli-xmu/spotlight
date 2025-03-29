train_script=train/llama2-7b-chat-spotlight.json

deepspeed \
    --include localhost:0,1,2,3,4,5,6,7 \
    train.py \
    --num_layers 32 \
    --max_tokens 4096 \
    --env_conf $train_script \
    --instance_per_cycle 4096 \
    --max_prepare_workers 8 \
    --prepare_batch_size_per_gpu 4 \
    --backward_per_head \
    --max_que 1024 \
    --max_top 1024 \
    --max_oth 1024 \
    --beta 1.0 \
    --margin 0.0 \
    --maskout 0.98 \
    --use_prepared_data

python train_results/convert.py --env_conf $train_script