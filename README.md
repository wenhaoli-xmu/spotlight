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

| Task              | Eval Command                   | Download Data Manually |
|-------------------|--------------------------------|------------------------|
| IoU               | bash scripts/test_iou.sh       | NO                     |
| Perplexity        | bash scripts/test_ppl.sh       | YES                    |
| Few-Shot Learning | bash scripts/test_lmeval.sh    | NO                     |
| LongBench         | bash scripts/test_longbench.sh | YES                    |
| Needle            | bash scripts/test_needle.sh    | YES                    |


**一次评测所有模型** 所有以 `test_` 开头的文件夹都对应着一种benchmark。且每个benchmark的bash脚本可以在 `scripts` 目录下找到，默认情况下，这些脚本会在所有base model（包括LLaMA2-7B, LLaMA2-7B-Chat, LLaMA3-8B）上评测所有的方法（包含Spotlight, Linear Hashing, Upper Bound, Quest, MagicPIG），时间较为漫长。

**仅仅评测某个模型** 如果想要仅仅运行某个base model的某种方法，例如只测试 LLaMA3-8B / w. Spotlight 在LongBench上的性能，则可以将 bash 脚本 `scripts/test_longbench.sh` 中的 `test_scripts` 变量修改为 `llama3-8b-spotlight.json` 。这里 `llama3-8b-spotlight.json` 的具体目录位于 `test_longbench/llama3-8b-spotlight.json`，里面定义了应该应该加载哪个checkpoint, 以及多少剪枝率等关键信息。

## <span id="eval-iou"> IoU </span>


## <span id="eval-ppl"> Perplexity </span>

1. 准备数据集.

    需要下载 [proof-pile.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/lm/proof-pile.json) 和 [codeparrot.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/lm/codeparrot.json) 两个文件，下载好之后，需要将两个文件的路径添加为环境变量:
    
    ```bash
    export SPOTLIGHT_PROOFPILE_PATH=/path/to/proof-pile.json
    export SPOTLIGHT_CODEPARROT_PATH=/path/to/code-parrot.json
    ```

2. 运行测试脚本.
    ```bash
    bash scripts/test_ppl.sh 
    ```


## <span id="eval-nlu"> Few-Shot Learning </span>

所有必要的数据集都会在脚本运行的时候自动下载：

```bash
bash scripts/test_lmeval.sh
```

注意，lm-eval-harness必须是0.3.0版本。

## <span id="eval-longbench"> LongBench </span>

1. 下载 [data.zip](https://huggingface.co/datasets/THUDM/LongBench/blob/main/data.zip)，完成之后将其拷贝到 `LongBench/data.zip` 

2. 测试在LongBench上的绝对分数
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

    **LLaMA2-7B-Chat**
    | Method      | Config      | Eval Log                                                                                                                     |
    |-------------|-------------|------------------------------------------------------------------------------------------------------------------------------|
    | Original    | N/A         | [llama2-7b-chat](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-chat.json)                           |
    | +Quest      | 1024 Budget | [llama2-7b-chat-quest-1024](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-chat-quest-1024.json)     |
    | +Quest      | 128 Budget  | [llama2-7b-chat-quest-128](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-chat-quest-128.json)       |
    | +MagicPIG   | Default     | [llama2-7b-chat-magicpig](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-chat-magicpig.json)         |
    | +Spotlight  | 90% Pruned  | [llama2-7b-chat-spotlight-90](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-chat-spotlight-90.json) |
    | +Spotilght  | 98% Pruned  | [llama2-7b-chat-spotlight-98](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama2-7b-chat-spotlight-98.json) |

    **LLaMA3-8B**
    | Method      | Config      | Eval Log                                                                                                           |
    |-------------|-------------|--------------------------------------------------------------------------------------------------------------------|
    | Original    | N/A         | [llama3-8b](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama3-8b.json)                           |
    | +Quest      | 1024 Budget | [llama3-8b-quest-1024](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama3-8b-quest-1024.json)     |
    | +Quest      | 256 Budget  | [llama3-8b-quest-256](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama3-8b-quest-256.json)       |
    | +MagicPIG   | Default     | [llama3-8b-magicpig](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama3-8b-magicpig.json)         |
    | +Spotlight  | 90% Pruned  | [llama3-8b-spotlight-90](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama3-8b-spotlight-90.json) |
    | +Spotilght  | 98% Pruned  | [llama3-8b-spotlight-98](https://github.com/wenhaoli-xmu/spotlight/test_longbench/log/llama3-8b-spotlight-98.json) |


## <span id="eval-fidelity"> Output Fidelity </span>

1. 


## <span id="eval-needle"> Needle In A Haystack </span>