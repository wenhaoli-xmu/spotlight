# Spotlight Attention: Exetreme KV Cache Pruning For LLM Generation

![img](spotlight.png)


# Shortcut

1. [Quick start](#quickstart)
2. [Model weights](#modelweights)
3. [Evaluation](#eval)
    
    * [Perplexity](#eval-ppl)
    * [Few-Shot Learning](#eval-nlu)
    * [LongBench](#eval-longbench)
    * [Output Fidelity](#eval-fidelity)
    * [Latency](#eval-latency)

4. [Train](#train)


# <span id="quickstart"> Quick Start </span>

```
git clone https://github.com/wenhaoli-xmu/spotlight    # training & evaluation
git clone https://github.com/wenhaoli-xmu/lsh-attn     # efficient kernels
git clone https://github.com/wenhaoli-xmu/lm-corpus    # training corpus
git clone https://github.com/wenhaoli-xmu/lm-profiler  # a tool for latency test

cd spotlight 
pip install -r requirement.txt
pip install -e .

cd ../lsh-attn
pip install -e .

cd ../corpus
pip install -e .

cd ../profiler
pip install -e .
```


# <span id="modelweights"> Model Weights </span>


# <span id="eval"> Evaluation </span>

| Task              | Eval Command                   | Download Data Manually |
|-------------------|--------------------------------|------------------------|
| IoU               | bash scripts/test_iou.sh       | NO                     |
| Perplexity        | bash scripts/test_ppl.sh       | YES                    |
| Few-Shot Learning | bash scripts/test_lmeval.sh    | NO                     |
| LongBench         | bash scripts/test_longbench.sh | YES                    |


**Evaluate across all base models and methods.**  Each dir starts with `test_` corresponding to a benchmark, and the launch scripts can be found in `scripts/`. By default, these scripts will run evaluation across all base models (including LLaMA2-7B, LLaMA2-7B-Chat and LLaMA3-8B), and all methods (including Spotlight, Linear Hashing, Upper Bound, Quest and MagicPIG). Thus it will take a long time to finish these evaluations.

**Evaluate single model with single method.**  If you want to just evaluate a single base model with a single modifying method, for example LLaMA3-8B with Spotlight Attention, you can edit the `test_scripts` variable in the shell script to the one you want, in this case, `llama3-8b-spotlight.json`. Here, the json file is located in the benchmark's dir, defining which checkpoint should be loaded as well as some configurations.

## <span id="eval-iou"> IoU </span>

1. Run test script
    ```bash
    bash scripts/test_iou.sh
    ```

2. For linear hashing, the shell script evaluate the training-free versionn. If you want to evaluate after training version, you can change the `load_ckp` keyword in the json file. For LLaMA2-7B as an example, you can change the `load_ckp` keyworkd in `test_iou/llama2-7b-linearhashing.json` to `ckp/llama2-7b-linearhashing.pth`. All related pth files can be found in [Model weights](#modelweights).


## <span id="eval-ppl"> Perplexity </span>

1. Prepare data.

    Download [proof-pile.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/lm/proof-pile.json) and [codeparrot.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/lm/codeparrot.json). After download, run the following command to set environment variables:
    
    ```bash
    export SPOTLIGHT_PROOFPILE_PATH=/path/to/proof-pile.json
    export SPOTLIGHT_CODEPARROT_PATH=/path/to/code-parrot.json
    ```

2. Run test scrtipt.
    ```bash
    bash scripts/test_ppl.sh 
    ```


## <span id="eval-nlu"> Few-Shot Learning </span>

All data required can be automatically downloaded on the fly when running this test script.

```bash
bash scripts/test_lmeval.sh
```

NOTE: the version of `lm-eval-harness` must be 0.3.0.

## <span id="eval-longbench"> LongBench </span>

1. Download [data.zip](https://huggingface.co/datasets/THUDM/LongBench/blob/main/data.zip), then copy it to `LongBench/data.zip`.

2. Test absolute score on LongBench
    ```bash
    bash scripts/test_longbench.sh
    ```
    Our evaluation results are provided:

    **LLaMA2-7B**
    | Method      | Config      | Eval Log                                                                                                           |
    |-------------|-------------|--------------------------------------------------------------------------------------------------------------------|
    | Original    | N/A         | [llama2-7b](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b.json)                           |
    | +Quest      | 1024 Budget | [llama2-7b-quest-1024](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-quest-1024.json)     |
    | +Quest      | 128 Budget  | [llama2-7b-quest-128](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-quest-128.json)       |
    | +MagicPIG   | Default     | [llama2-7b-magicpig](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-magicpig.json)         |
    | +Spotlight  | 90% Pruned  | [llama2-7b-spotlight-90](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-spotlight-90.json) |
    | +Spotilght  | 98% Pruned  | [llama2-7b-spotlight-98](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-spotlight-98.json) |

    **LLaMA2-7B-Chat**
    | Method      | Config      | Eval Log                                                                                                                     |
    |-------------|-------------|------------------------------------------------------------------------------------------------------------------------------|
    | Original    | N/A         | [llama2-7b-chat](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-chat.json)                           |
    | +Quest      | 1024 Budget | [llama2-7b-chat-quest-1024](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-chat-quest-1024.json)     |
    | +Quest      | 128 Budget  | [llama2-7b-chat-quest-128](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-chat-quest-128.json)       |
    | +MagicPIG   | Default     | [llama2-7b-chat-magicpig](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-chat-magicpig.json)         |
    | +Spotlight  | 90% Pruned  | [llama2-7b-chat-spotlight-90](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-chat-spotlight-90.json) |
    | +Spotilght  | 98% Pruned  | [llama2-7b-chat-spotlight-98](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama2-7b-chat-spotlight-98.json) |

    **LLaMA3-8B**
    | Method      | Config      | Eval Log                                                                                                           |
    |-------------|-------------|--------------------------------------------------------------------------------------------------------------------|
    | Original    | N/A         | [llama3-8b](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama3-8b.json)                           |
    | +Quest      | 1024 Budget | [llama3-8b-quest-1024](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama3-8b-quest-1024.json)     |
    | +Quest      | 256 Budget  | [llama3-8b-quest-256](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama3-8b-quest-256.json)       |
    | +MagicPIG   | Default     | [llama3-8b-magicpig](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama3-8b-magicpig.json)         |
    | +Spotlight  | 90% Pruned  | [llama3-8b-spotlight-90](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama3-8b-spotlight-90.json) |
    | +Spotilght  | 98% Pruned  | [llama3-8b-spotlight-98](https://github.com/wenhaoli-xmu/spotlight/tree/main/test_longbench/log/llama3-8b-spotlight-98.json) |


## <span id="eval-fidelity"> Output Fidelity </span>

1. The LongBench output files are required to test output fidelity. You can either get these output files by runing the test scripts in yourself, or use our provided ones. For example, if you want to evaluate the output similarity between LLaMA3-8B model with and without Spotlight Attention, you can run the following command:

    ```bash
    python test_longbench/test_sim.py test_longbench/log/llama3-8b.json test_longbench/log/llaam3-8b-spotlight.json
    ```


## <span id="eval-latency"> Latency </span>

1. Entry `lsh-attn` directorty
    
    There are some python files in this directory, which are used to test the lantecy of several efficient attention kernel as well as some component.
    ```
    ├── benchmark_flashattn.py
    ├── benchmark_flashinfer.py
    ├── benchmark_pack.py
    ├── benchmark_quest.py
    ├── benchmark_sdpa.py
    ├── benchmark_spotlight.py
    ```

2. Install [triton](https://triton-lang.org)，[flash attention 2.5.8](https://github.com/Dao-AILab/flash-attention/releases)，and [flash infer 0.1.6](https://docs.flashinfer.ai)

3. Run the following command to test latency.
    ```bash
    python benchmark_flashattn.py
    python benchmark_flashinfer.py
    python benchmark_pack.py
    python benchmark_quest.py
    python benchmark_sdpa.py
    python benchmark_spotlight.py
    ```

    The batch size are set to 1 by default, you can change it by providing additional arguments `--batch_size 4`.
    

# <span id="train"> Train </span>

1. Create directories
    ```bash
    cd spotlight
    mkdir -p data/slimpajama
    mkdir ckp
    ```

2. Download [arxiv.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/slimpajama/arxiv.json) and [book.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/slimpajama/book.json), then put them under `data/slimpajama`.

3. Training can be executed in two ways.
    * **When the disk has enough room left**，you can storage the activations every layer only once before training.
        
        Specifically, edit the `train.sh` script with addtiional argument `--prepare_data`, then run it. After completing, change `--prepare_data` to `--use_prepared_data` and run it again.

    * **When the disk has little space**, you can generate these activations and use them on the fly.

        This is our defualt traininng method, you can just run the training script without modifying anything.

    
    After training, the checkpoint file can be found under `ckp`, and can be used in the `load_ckp` keyword in all test scripts.

4. Memory reuduction tricks.

    In the training process, the calculation of ranking loss cost tons of memory, because the size of tensor $Z$ is extra large: 
    $$
    n_{heads}\times n_{query}\times n_{top} \times (n_{query} - n_{top})
    $$

    To this end, we use some tricks to mitigate this issue. The most effective one is to restrict the number of tokens involved, controled by three arguments: `--max_que`, `--max_top` and `--max_oth`. By default, they are set to 1024 and we recommand you to shrink `--max_que` and `--max_oth` first when you encounter OOM issue.
