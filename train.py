import torch
import gc
import os

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from functools import partial
import tqdm
import argparse

from concurrent.futures import ThreadPoolExecutor
import concurrent

from spotlight import get_monkey_patch
from spotlight.misc import adjust_lr

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch.distributed as dist
import json


class TrainingData(torch.utils.data.Dataset):
    def __init__(self, args):
        self.data = []
        for data in args.train_data:
            with open(data, 'r') as f:
                for line in f:
                    self.data.append(json.loads(line))
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def compute_attn_supervise_loss(
        draft_attn, 
        true_attn, 
        random_query_index, 
        max_top, 
        max_oth, 
        maskout):

    # prepare & apply mask
    num_kv = true_attn.shape[-1]
    mask = torch.triu(torch.ones((num_kv, num_kv), dtype=torch.bool, device=true_attn.device), diagonal=1)[None, None, :, :]
    
    if random_query_index is not None:
        mask = mask[..., random_query_index, :]
    
    true_attn = torch.masked_fill(true_attn, mask, value=torch.finfo(true_attn.dtype).min)

    indices = torch.argsort(true_attn, dim=-1, descending=True)

    top_cnt = int(indices.shape[-1] * (1 - maskout))
    top_indices = indices[..., :top_cnt]
    oth_indices = indices[..., top_cnt:]

    if max_top is not None:
        top_rnd_indices = torch.randperm(top_cnt, dtype=torch.int64, device=indices.device)[:max_top]
        top_indices = top_indices[..., top_rnd_indices]
    if max_oth is not None:
        oth_rnd_indices = torch.randperm(indices.shape[-1] - top_cnt, dtype=torch.int64, device=indices.device)[:max_oth]
        oth_indices = oth_indices[..., oth_rnd_indices]

    top_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=top_indices)[..., :, None]
    oth_mask = torch.gather(mask.expand_as(true_attn), dim=-1, index=oth_indices)[..., None, :]

    top_attn = torch.gather(draft_attn, dim=-1, index=top_indices)[..., :, None]
    oth_attn = torch.gather(draft_attn, dim=-1, index=oth_indices)[..., None, :]

    zeros = top_attn.new_zeros(top_attn.shape[:-1] + oth_attn.shape[-1:]).flatten(-2)
    top_attn = top_attn.expand(-1, -1, -1, -1, top_attn.shape[-2])
    oth_attn = oth_attn.expand(-1, -1, -1, oth_attn.shape[-1], -1)

    top_attn = top_attn.flatten(-2)
    oth_attn = oth_attn.flatten(-2)
    union_mask = (top_mask | oth_mask).flatten(-2)

    zeros = zeros[~union_mask.bool()]
    top_attn = top_attn[~union_mask.bool()]
    oth_attn = oth_attn[~union_mask.bool()]

    top_acc = (top_attn > 0).count_nonzero().item() / top_attn.numel()
    oth_acc = (oth_attn < 0).count_nonzero().item() / oth_attn.numel()

    logits = torch.stack([zeros, -top_attn, oth_attn], dim=0)
    loss = torch.logsumexp(logits, dim=0).mean(-1).sum()

    return top_acc, oth_acc, loss


def get_optimizer_and_lr_adjuster(max_lr, train_iters, warmup, weight_decay, beta1, beta2, params, **kwargs):
    optim = torch.optim.AdamW(params, lr=max_lr, betas=[beta1, beta2], weight_decay=weight_decay)
    lr_adjuster = partial(adjust_lr, optim=optim, total=train_iters, max_lr=max_lr, min_lr=0, restart=1, warmup=warmup, plateau=0)
    return optim, lr_adjuster


def clear_cache():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()


def collate_fn(batch, pad_token_id, max_tokens):
    if pad_token_id is None:
        pad_token_id = 0

    input_ids = [x.get('input_ids') for x in batch]
    input_len = [len(x) for x in input_ids]

    assert all([length <= max_tokens for length in input_len]), f"Input length exceed `max_tokens`, please enlarge `max_tokens` or shrink truncation length to fix the problem."

    # padding
    input_ids = [x + [pad_token_id] * (max_tokens - len(x)) for x in input_ids]

    # to tensor
    input_ids = torch.tensor(input_ids, dtype=torch.int64)
    input_len = torch.tensor(input_len, dtype=torch.int64)

    return {"input_ids": input_ids.cuda(), "input_len": input_len}


def reset_buffer_dir(buffer, disable):
    if not disable:
        if dist.get_rank() == 0:
            for file in os.listdir(buffer):
                os.remove(os.path.join(buffer, file))
        dist.barrier()


def train(args):
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group('nccl', rank=local_rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    num_gpus = dist.get_world_size()
    os.makedirs(args.buffer, exist_ok=True)
    reset_buffer_dir(args.buffer, args.use_prepared_data)

    assert args.train_iters % args.instance_per_cycle == 0
    assert args.num_layers % num_gpus == 0
    assert args.instance_per_cycle % args.prepare_batch_size_per_gpu == 0
    assert (args.instance_per_cycle // args.prepare_batch_size_per_gpu) % num_gpus == 0

    num_inn_cycle = args.train_iters // args.instance_per_cycle
    num_out_cycle = args.num_layers // num_gpus
    gen_range = [args.out_cycle] if args.out_cycle is not None else range(num_out_cycle)
    executor = ThreadPoolExecutor(max_workers=args.max_prepare_workers)
    monkey_patch = get_monkey_patch(args.method)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    pad_token_id = tokenizer.pad_token_id
    del tokenizer

    # start training
    for out_cycle_idx in gen_range:

        torch.manual_seed(42)

        # skip layers that were already been trained
        pth_template = "train_results/{}/{}.pth"
        pth_exist = lambda layer_idx: os.path.exists(
            pth_template.format(
                args.model_name_or_path.split('/')[-1].lower(), 
                out_cycle_idx * num_gpus + layer_idx))

        if all([pth_exist(layer_idx) for layer_idx in range(num_gpus)]):
            continue

        # load model and tokenizer
        layer_idx = out_cycle_idx * num_gpus + local_rank
        layer_indices = [out_cycle_idx * num_gpus + i for i in range(num_gpus)]

        # create directories
        save_path = args.model_name_or_path.split('/')[-1].lower()
        os.makedirs(f"train_results/{save_path}", exist_ok=True)

        # skip
        local_cond = torch.tensor(os.path.exists(f"train_results/{save_path}/history-{layer_idx}.pth"), dtype=bool, device='cuda')
        global_cond = [torch.zeros_like(local_cond) for _ in range(dist.get_world_size())]
        dist.all_gather(global_cond, local_cond)
        if global_cond.all():
            print(f"Skipping layer-{layer_idx} because train results already exist.")
            continue

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map='cuda',
            torch_dtype=torch.bfloat16)
        model = monkey_patch(model)
            
        # delete other layers except the current training one
        layer = model.dump_as_attn_modules()[layer_idx]
        layer.train()
        params = list(layer.query_hash.parameters()) + list(layer.key_hash.parameters())

        # get position embeddings
        pos_ids = torch.arange(model.config.max_position_embeddings, dtype=torch.long, device='cuda')
        pos_ids = pos_ids.unsqueeze(0)
        
        with torch.no_grad():
            fake_tensor = torch.empty(1, dtype=torch.bfloat16, device='cuda')
            pos_emb = model.model.rotary_emb(fake_tensor, pos_ids)

        del model

        # clear cache after model deleted
        clear_cache()
        print(f"RANK-{local_rank} training started !")
        dist.barrier()

        # construct dataloader
        if not args.use_prepared_data:
            corpus = TrainingData(args)
            partial_collate_fn = partial(
                collate_fn, 
                pad_token_id=pad_token_id, 
                max_tokens=args.max_tokens)
            sampler = DistributedSampler(
                corpus, 
                num_replicas=num_gpus, 
                rank=local_rank, 
                shuffle=True)
            loader = DataLoader(
                corpus, 
                batch_size=args.prepare_batch_size_per_gpu, 
                sampler=sampler,
                collate_fn=partial_collate_fn)
            data_iter = iter(loader)
            sampler.set_epoch(0)

        # construct optimizer and learning rate adjuster
        optim, lr_adjust = get_optimizer_and_lr_adjuster(
            args.max_lr,
            args.train_iters,
            args.warmup,
            args.weight_decay,
            args.beta1,
            args.beta2,
            params=params)
        
        step = 0

        history_loss = []
        history_tacc = []
        history_oacc = []

        compute_loss = partial(
            compute_attn_supervise_loss,
            max_top=args.max_top, 
            max_oth=args.max_oth,
            maskout=args.maskout)

        for _ in range(num_inn_cycle):

            if not args.use_prepared_data:

                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.bfloat16,
                    device_map='cuda')
                model = monkey_patch(model)
                model.eval()
                dist.barrier()

                increment = num_gpus * args.prepare_batch_size_per_gpu
                futures = []

                for idx in tqdm.tqdm(range(0, args.instance_per_cycle, increment)):

                    inputs = next(data_iter)
                    length = inputs.get("input_len")
                    inputs.update({"return_hidden_states": True})

                    # forward pass
                    with torch.no_grad():
                        outputs = model(**inputs)
                    inputs = [outputs.hidden_states[i].cuda(local_rank) for i in layer_indices]
                    inputs = torch.stack(inputs, dim=0)

                    # inter-process communication-1, exchange the hidden states
                    inputs_gather = [torch.empty_like(inputs) for _ in range(num_gpus)]
                    dist.all_gather(inputs_gather, inputs)
                    inputs_gather = [inputs[local_rank] for inputs in inputs_gather]
                    inputs_gather = torch.cat(inputs_gather, dim=0).cpu()

                    # inter-process communication-2, exchange the input sequence length
                    length = torch.tensor(length, dtype=torch.int64, device=local_rank)
                    length_gather = [torch.empty_like(length) for _ in range(num_gpus)]
                    dist.all_gather(length_gather, length)
                    length_gather = torch.cat(length_gather)

                    # save data
                    buffer = (inputs_gather, length_gather)
                    buffer_file = f"{args.buffer}/inputs_buffer_rank_{local_rank}_{idx:05d}.pt"

                    future = executor.submit(torch.save, buffer, buffer_file)
                    futures.append(future)
                    if len(futures) >= args.max_prepare_workers:
                        concurrent.futures.wait(futures)
                        futures = []

                # wait for all process to finish data preprocessing
                del model, inputs, outputs
                clear_cache()
                dist.barrier()

                if args.prepare_data:
                    print(f"RANK-{local_rank} data preparation finished.")
                    return

            # sort the buffer files according to their IDs
            buffer_files = os.listdir(args.buffer)
            buffer_files = sorted(filter(
                lambda x: x.startswith(f"inputs_buffer_rank_{local_rank}"), 
                buffer_files))

            # read the first buffer file
            inputs_gather, length_gather = torch.load(os.path.join(args.buffer, buffer_files[0]))
            buffer_files = [*buffer_files[1:], buffer_files[0]]

            # traverse remaining buffer files
            for buffer_file in buffer_files:

                # prefetch next file
                future = executor.submit(torch.load, os.path.join(args.buffer, buffer_file))

                for hidden_states, length in zip(inputs_gather, length_gather):
                    hidden_states = hidden_states[:length, ...].unsqueeze(0)
                    lr_adjust(step=step)

                    # forward & backward
                    random_query_index = None
                    if args.max_que is not None:
                        random_query_index = torch.randperm(
                            hidden_states.shape[-2], 
                            dtype=torch.int64)[:args.max_que].cuda()

                    _, (draft_attn, true_attn) = layer(
                        hidden_states=hidden_states.to(local_rank), 
                        position_embeddings=pos_emb,
                        random_query_index=random_query_index)

                    if args.backward_per_head:
                        grad = torch.zeros_like(true_attn)
                        tmp_loss = []
                        tmp_tacc = []
                        tmp_oacc = []

                        # per head calculation to save GPU memory
                        for head_idx, (draft_attn_head, true_attn_head) in enumerate(zip(
                            torch.chunk(draft_attn, chunks=draft_attn.shape[1], dim=1),
                            torch.chunk(true_attn, chunks=draft_attn.shape[1], dim=1),
                        )):
                            draft_attn_head = draft_attn_head.detach()
                            true_attn_head = true_attn_head.detach()
                            draft_attn_head.requires_grad_(True)

                            top_acc, oth_acc, loss = compute_loss(draft_attn_head, true_attn_head, random_query_index)
                            loss.backward()

                            grad[:, head_idx, ...] = draft_attn_head.grad.data[:]

                            tmp_loss.append(loss.item())
                            tmp_tacc.append(top_acc)
                            tmp_oacc.append(oth_acc)

                        history_loss.append(sum(tmp_loss))
                        history_tacc.append(sum(tmp_tacc) / len(tmp_tacc))
                        history_oacc.append(sum(tmp_oacc) / len(tmp_oacc))

                        grad /= args.gradient_accumulation
                        draft_attn.backward(gradient=grad)
                        del grad, true_attn_head, draft_attn_head

                    else:
                        # direct calculation (NOTE: will cause mask selection error in training 8192 length models)
                        top_acc, oth_acc, loss = compute_loss(draft_attn, true_attn, random_query_index)
                        
                        history_loss.append(loss.item())
                        history_tacc.append(top_acc)
                        history_oacc.append(oth_acc)
                        
                        loss /= args.gradient_accumulation
                        loss.backward()

                    # output key informantion
                    print(f"layer: {layer_idx:>5d} | "
                        f"step: {step:>5d} | "
                        f"loss: {history_loss[-1]:>05.3f} | "
                        f"tacc: {history_tacc[-1]:>05.3f} | "
                        f"oacc: {history_oacc[-1]:>05.3f}", flush=True)

                    # update the parameters
                    if (step + 1) % args.gradient_accumulation == 0:
                        if args.gradient_clipping is not None:
                            torch.nn.utils.clip_grad_norm_(params, max_norm=args.gradient_clipping)
                        optim.step()
                        optim.zero_grad()

                    step += 1

                    del draft_attn, true_attn, hidden_states, loss
                    clear_cache()
                    dist.barrier()

                # get the already prefetched data
                inputs_gather, length_gather = future.result()

            reset_buffer_dir(args.buffer, args.use_prepared_data)

        # save layerwise weight files
        info = {"loss": history_loss, "tacc": history_tacc, "oacc": history_oacc}
        torch.save(info, f"train_results/{save_path}/history-{layer_idx}.pth")
        torch.save(list(params), f"train_results/{save_path}/weight-{layer_idx}.pth")
        print(f"RANK-{local_rank} training done !")
        dist.barrier()

        # clear buffer
        del layer, optim
        clear_cache()
        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model related parameters (NOTE: need to change according to the base model)
    parser.add_argument("--num_layers", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--maskout", type=float, default=0.98)
    parser.add_argument("--model-name-or-path", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)

    # model non-related parameters
    parser.add_argument("--prepare_data", action='store_true')
    parser.add_argument("--use_prepared_data", action='store_true')
    parser.add_argument("--buffer", type=str, default='train_buffer')
    parser.add_argument("--out_cycle", type=int, default=None)

    # resource controling related parameters
    parser.add_argument("--instance_per_cycle", type=int, default=1000)
    parser.add_argument("--prepare_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--max_prepare_workers", type=int, default=4)
    parser.add_argument("--max_top", type=int, default=None)
    parser.add_argument("--max_oth", type=int, default=None)
    parser.add_argument("--max_que", type=int, default=None)
    parser.add_argument("--backward_per_head", action='store_true')

    # training data
    parser.add_argument("--train-data", type=str, action='append')
    parser.add_argument("--train-iters", type=int)

    # adamw configuration
    parser.add_argument("--max-lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=float, default=0.)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.98)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-clipping", type=float, default=None)
    args = parser.parse_args()
    train(args)
