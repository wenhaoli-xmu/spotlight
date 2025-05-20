train_script=/mnt/petrelfs/lijie/spotlight/train/qwen2.5-3b-spotlight.json


deepspeed \
    --include localhost:0,1,2,3 \
    train.py \
    --num_layers 36 \
    --max_tokens 8192 \
    --env_conf $train_script \
    --instance_per_cycle 144 \
    --max_prepare_workers 8 \
    --prepare_batch_size_per_gpu 4 \
    --max_que 1024 \
    --max_top 1024 \
    --max_oth 1024 \
    --backward_per_head \
    --beta 1.0 \
    --margin 0.0 \
    --maskout 0.98 \
    --buffer train_buffer_qwen_3b

python train_results/convert.py --env_conf $train_script --num_layers 36