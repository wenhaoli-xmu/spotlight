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
    * [Needle In A Haystack](#eval-needle)
    * [Latency](#eval-latency)

4. [Training](#train)
5. [Acknowledgement](#acknowledge)


# <span id="quickstart"> Quick Start </span>

**Step1** Clone the following repositories.
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

## <span id="eval-ppl"> Perplexity </span>

1. Download Proof-Pile and CodeParrot datasets.
    **Comming Soon**

2. Run the test script.
    ```bash
    bash scripts/test_ppl.sh 
    ```


## <span id="eval-nlu"> Few-Shot Learning </span>

```bash
bash scripts/test_lmeval.sh
```


## <span id="eval-longbench"> LongBench </span>

1. 下载LongBench评测需要的数据集
    **Comming Soon**

2. 运行测试脚本
    ```bash
    bash scripts/test_longbench.sh
    ```
    我们自己的运行结果如下

    **LLaMA2-7B**
    | Method      | Config      | Eval Log                                                                                                           |
    |-------------|-------------|--------------------------------------------------------------------------------------------------------------------|
    | Original    | N/A         | [llama2-7b](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b.json)                           |
    | +Quest      | 1024 Budget | [llama2-7b-quest-1024](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-quest-1024.json)     |
    | +Quest      | 128 Budget  | [llama2-7b-quest-128](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-quest-128.json)       |
    | +MagicPIG   | Default     | [llama2-7b-magicpig](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-magicpig.json)         |
    | +Spotlight  | 90% Pruned  | [llama2-7b-spotlight-90](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-spotlight-90.json) |
    | +Spotilght  | 98% Pruned  | [llama2-7b-spotlight-98](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-spotlight-98.json) |

## <span id="eval-fidelity"> Output Fidelity </span>

1. 


## <span id="eval-needle"> Needle In A Haystack </span>