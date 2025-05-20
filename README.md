# Spotlight Attention: Exetreme KV Cache Pruning For LLM Generation

![img](spotlight.png)


# Shortcut

1. [Installation](#install)
2. [Model weights](#modelweights)
3. [Evaluation](#eval)
    
    * [Perplexity](#eval-ppl)
    * [Few-Shot Learning](#eval-nlu)
    * [LongBench](#eval-longbench)
    * [Output Fidelity](#eval-fidelity)
    * [Latency](#eval-latency)

4. [Train](#train)


# <span id="install"> Installation </span>

```
git clone https://anonymous.4open.science/r/spotlight-new    # training & evaluation
git clone https://anonymous.4open.science/r/lm-corpus-FAB7    # training corpus
git clone https://anonymous.4open.science/r/lm-profiler-A550  # a tool for latency test

cd spotlight 
pip install -r requirement.txt
pip install -e .

cd ../corpus
pip install -e .

cd ../profiler
pip install -e .
```

CUDA kernel installation (optional)
```
cd spotlight/spotlight/kernel
bash install.sh

# After successful compilation, two more `*.so` files will be added to the kernel directory.
```


# <span id="modelweights"> Model Weights </span>




# <span id="eval"> Evaluation </span>

| Task              | Eval Command                   | Download Data Manually |
|-------------------|--------------------------------|------------------------|
| IoU               | bash scripts/test_iou.sh       | NO                     |
| Perplexity        | bash scripts/test_ppl.sh       | YES                    |
| Few-Shot Learning | bash scripts/test_lmeval.sh    | NO                     |
| LongBench         | bash scripts/test_longbench.sh | YES                    |


**Parallel Execution**  Each directory prefixed with `test_` corresponds to a specific benchmark, and the associated launch scripts are located within the `scripts/` directory. By default, these scripts are designed to execute evaluations across all base models (including LLaMA2-7B, LLaMA2-7B-Chat, and LLaMA3-8B) and all methods (such as Spotlight, Linear Hashing, Upper Bound, Quest, and MagicPIG). Completing these evaluations may require a significant amount of time.

**Selective Evaluation**  To streamline the process and focus on evaluating a specific base model with a particular modification method—for instance, LLaMA3-8B using Spotlight Attention—you can modify the `test_scripts` variable within the shell script. Specifically, set it to the desired configuration file, such as `llama3-8b-spotlight.json`. This JSON file, located in the benchmark's directory, specifies the checkpoint to be loaded and includes various configuration settings tailored to the evaluation.

## <span id="eval-iou"> IoU </span>

1. Execute the following command to run the test script:
    ```bash
    bash scripts/test_iou.sh
    ```

2. By default, the shell script evaluates the training-free version of linear hashing. If you wish to evaluate the trained version instead, modify the `load_ckp` keyword in the corresponding JSON file. For example, to evaluate the trained version for LLaMA2-7B, update the `load_ckp` key in `test_iou/llama2-7b-linearhashing.json` to point to the trained checkpoint file, such as `ckp/llama2-7b-linearhashing.pth`. All relevant `.pth` files are available in the [Model weights](#modelweights) section.


## <span id="eval-ppl"> Perplexity </span>

1. Prepare data.

    Download the required datasets: [proof-pile.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/lm/proof-pile.json) and [codeparrot.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/lm/codeparrot.json). After downloading, set the environment variables by running the following commands:
    
    ```bash
    export SPOTLIGHT_PROOFPILE_PATH=/path/to/proof-pile.json
    export SPOTLIGHT_CODEPARROT_PATH=/path/to/code-parrot.json
    ```

    Replace `/path/to/` with the actual paths to the downloaded files.

2. Execute the test script with the following command:
    ```bash
    bash scripts/test_ppl.sh 
    ```


## <span id="eval-nlu"> Few-Shot Learning </span>

All necessary data will be automatically downloaded during the execution of the test script. Simply run the following command:

```bash
bash scripts/test_lmeval.sh
```

NOTE: Ensure that the version of `lm-eval-harness` is 0.3.0.

## <span id="eval-longbench"> LongBench </span>

1. Download [data.zip](https://huggingface.co/datasets/THUDM/LongBench/blob/main/data.zip) and place it in the `LongBench/` directory, ensuring it is named `LongBench/data.zip`.

2. Run the following command to test the absolute scores:
    ```bash
    bash scripts/test_longbench.sh
    ```
    Our evaluation results are provided below:

    **LLaMA2-7B**
    | Method      | Config      | Eval Log                                                                                                           |
    |-------------|-------------|--------------------------------------------------------------------------------------------------------------------|
    | Original    | N/A         | [llama2-7b](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b.json)                           |
    | +Quest      | 1024 Budget | [llama2-7b-quest-1024](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-quest-1024.json)     |
    | +Quest      | 128 Budget  | [llama2-7b-quest-128](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-quest-128.json)       |
    | +MagicPIG   | Default     | [llama2-7b-magicpig](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-magicpig.json)         |
    | +Spotlight  | 90% Pruned  | [llama2-7b-spotlight-90](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-spotlight-90.json) |
    | +Spotilght  | 98% Pruned  | [llama2-7b-spotlight-98](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-spotlight-98.json) |

    **LLaMA2-7B-Chat**
    | Method      | Config      | Eval Log                                                                                                                     |
    |-------------|-------------|------------------------------------------------------------------------------------------------------------------------------|
    | Original    | N/A         | [llama2-7b-chat](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-chat.json)                           |
    | +Quest      | 1024 Budget | [llama2-7b-chat-quest-1024](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-chat-quest-1024.json)     |
    | +Quest      | 128 Budget  | [llama2-7b-chat-quest-128](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-chat-quest-128.json)       |
    | +MagicPIG   | Default     | [llama2-7b-chat-magicpig](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-chat-magicpig.json)         |
    | +Spotlight  | 90% Pruned  | [llama2-7b-chat-spotlight-90](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-chat-spotlight-90.json) |
    | +Spotilght  | 98% Pruned  | [llama2-7b-chat-spotlight-98](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama2-7b-chat-spotlight-98.json) |

    **LLaMA3-8B**
    | Method      | Config      | Eval Log                                                                                                           |
    |-------------|-------------|--------------------------------------------------------------------------------------------------------------------|
    | Original    | N/A         | [llama3-8b](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama3-8b.json)                           |
    | +Quest      | 1024 Budget | [llama3-8b-quest-1024](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama3-8b-quest-1024.json)     |
    | +Quest      | 256 Budget  | [llama3-8b-quest-256](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama3-8b-quest-256.json)       |
    | +MagicPIG   | Default     | [llama3-8b-magicpig](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama3-8b-magicpig.json)         |
    | +Spotlight  | 90% Pruned  | [llama3-8b-spotlight-90](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama3-8b-spotlight-90.json) |
    | +Spotilght  | 98% Pruned  | [llama3-8b-spotlight-98](https://anonymous.4open.science/r/spotlight-new/test_longbench/log/llama3-8b-spotlight-98.json) |


## <span id="eval-fidelity"> Output Fidelity </span>

1. Prerequisites:
    To evaluate output fidelity, you need the LongBench output files. These files can be obtained in one of two ways:
   
    * Run the test scripts yourself to generate the output files.
    * Use the output files we provide.
  
   For example, to evaluate the output similarity between the LLaMA3-8B model with and without Spotlight Attention, execute the following command:

    ```bash
    python test_longbench/test_sim.py test_longbench/log/llama3-8b.json test_longbench/log/llaam3-8b-spotlight.json
    ```


## <span id="eval-latency"> Latency </span>

1. Navigate to the `lsh-attn` Directory
    
    This directory contains Python scripts for testing the latency of various efficient attention kernels and components:
    ```
    ├── benchmark_flashattn.py
    ├── benchmark_flashinfer.py
    ├── benchmark_pack.py
    ├── benchmark_quest.py
    ├── benchmark_sdpa.py
    ├── benchmark_spotlight.py
    ```

2. Install Dependencies
   Ensure the following dependencies are installed:

    * [triton](https://triton-lang.org)
    * [flash attention 2.5.8](https://github.com/Dao-AILab/flash-attention/releases)
    * [flash infer 0.1.6](https://docs.flashinfer.ai)

4. Run Latency Tests
   Execute the following commands to test the latency of each component:
    ```bash
    python benchmark_flashattn.py
    python benchmark_flashinfer.py
    python benchmark_pack.py
    python benchmark_quest.py
    python benchmark_sdpa.py
    python benchmark_spotlight.py
    ```

    By default, the batch size is set to 1. You can modify it by adding the `--batch_size` argument, for example:

    ```python
    python benchmark_spotlight.py --batch_size 4
    ```
    

# <span id="train"> Train </span>

1. Create Directories
    ```bash
    cd spotlight
    mkdir -p data/slimpajama
    mkdir ckp
    ```

2. Download Datasets
   Download the required datasets:
   
   * [arxiv.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/slimpajama/arxiv.json)
   * [book.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/slimpajama/book.json)
   
   Place these files under the `data/slimpajama` directory.

3. Training Execution
   Training can be performed in two ways, depending on available disk space:

   * When disk space is sufficient. Store the activations for each layer once before training.

       1. Edit the `train.sh` script by adding the `--prepare_data` argument and run it.
       2. After completion, replacing `--prepare_data` with `--use_prepared_data` and run the script again.
    
   * When disk space is limited. Generate and use activations on the fly. This is the default training method. Simply run the training script without any modifications.
  
   After training, the checkpoint file will be saved under the `ckp` directory. This checkpoint can be referenced using the `load_ckp` keyword in all test scripts.

5. Memory Reduction Techniques.

   During training, calculating the ranking loss consumes significant memory due to the large size of tensor $Z$:
   
    $$
    n_{heads}\times n_{query}\times n_{top} \times (n_{query} - n_{top})
    $$

   To address this, we employ several memory-saving techniques. The most effictive approach is to limit the number of tokens involved, controlled by the following arguments:

   * `--max_que`
   * `--max_top`
   * `--max_oth`
  
   By default, these are set to 1024. If you encounter out-of-memory (OOM) issues, we recommand reducing `--max_que` and `--max_oth` first.
