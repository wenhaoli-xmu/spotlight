# Spotlight Attention: Extreme KV Cache Pruning for LLM Generation

![Spotlight Attention](spotlight.png)

---

## Table of Contents

1. [Installation](#installation)
2. [Model Weights](#model-weights)
3. [Evaluation](#evaluation)
   - [IoU](#eval-iou)
   - [Perplexity](#eval-perplexity)
   - [Few-Shot Learning](#eval-few-shot)
   - [LongBench](#eval-longbench)
   - [QA Response Fidelity](#eval-fidelity)
4. [Training](#training)

---

## <span id="installation"> 🚀 Installation </span>

Clone the necessary repositories and install dependencies:

```bash
# Clone repositories
git clone https://anonymous.4open.science/r/spotlight       # Training and evaluation
git clone https://anonymous.4open.science/r/lm-corpus-FAB7     # Training corpus
git clone https://anonymous.4open.science/r/lm-profiler-A550   # Latency testing tool

# Install dependencies
cd spotlight
pip install -r requirements.txt
pip install -e .

cd ../lm-corpus-FAB7
pip install -e .

cd ../lm-profiler-A550
pip install -e .
```

### Optional: CUDA Kernel Installation

For enhanced performance, install the CUDA kernel:

```bash
cd spotlight/spotlight/kernel
bash install.sh
```

Upon successful compilation, two `.so` files will be added to the `kernel` directory.

---

## <span id="model-weights"> 💾 Model Weights </span>

Pre-trained model checkpoints are available for download:

| Model           | Checkpoint                                                                 |
|-----------------|---------------------------------------------------------------------------|
| LLaMA3-8B       | [llama3-8b-spotlight.pth](https://anonymous.4open.science/r/spotlight/ckp/llama3-8b-spotlight.pth)       |
| LLaMA3-8B (C4)  | [llama3-8b-spotlight-c4.pth](https://anonymous.4open.science/r/spotlight/ckp/llama3-8b-spotlight-c4.pth) |
| LLaMA3-8B (Code)| [llama3-8b-spotlight-code.pth](https://anonymous.4open.science/r/spotlight/ckp/llama3-8b-spotlight-code.pth) |
| Qwen2.5-1.5B    | [qwen2.5-1.5b-spotlight.pth](https://anonymous.4open.science/r/spotlight/ckp/qwen2.5-1.5b-spotlight.pth) |
| Qwen2.5-3B      | [qwen2.5-3b-spotlight.pth](https://anonymous.4open.science/r/spotlight/ckp/qwen2.5-3b-spotlight.pth)     |
| Qwen2.5-7B      | [qwen2.5-7b-spotlight.pth](https://anonymous.4open.science/r/spotlight/ckp/qwen2.5-7b-spotlight.pth)     |
| Qwen2.5-14B     | [qwen2.5-14b-spotlight.pth](https://anonymous.4open.science/r/spotlight/ckp/qwen2.5-14b-spotlight.pth)   |

---

## <span id="evaluation"> 📊 Evaluation </span>

### <span id="eval-iou"> IoU </span>

Evaluate the Intersection over Union (IoU) metric:

1. Run the test script:

   ```bash
   bash scripts/test_iou.sh
   ```

2. By default, the script evaluates the training-free linear hashing version. To evaluate a trained model, update the `load_ckp` key in the relevant JSON configuration file (e.g., `test_iou/llama2-7b-linearhashing.json`) to point to the desired checkpoint from the [Model Weights](#model-weights) section.

### <span id="eval-perplexity"> Perplexity </span>

1. **Prepare Datasets**:

   Download the required datasets:
   - [proof-pile.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/lm/proof-pile.json)
   - [codeparrot.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/lm/codeparrot.json)

   Set environment variables:

   ```bash
   export SPOTLIGHT_PROOFPILE_PATH=/path/to/proof-pile.json
   export SPOTLIGHT_CODEPARROT_PATH=/path/to/codeparrot.json
   ```

   Replace `/path/to/` with the actual file paths.

2. **Run Evaluation**:

   ```bash
   bash scripts/test_ppl.sh
   ```

### <span id="eval-few-shot"> Few-Shot Learning </span>

All required datasets are automatically downloaded during evaluation. Ensure `lm-eval-harness` version 0.3.0 is installed, then run:

```bash
bash scripts/test_lmeval.sh
```

### <span id="eval-longbench"> LongBench </span>

1. **Prepare Datasets**:

   Download [data.zip](https://huggingface.co/datasets/THUDM/LongBench/blob/main/data.zip) and place it in the `LongBench/` directory as `LongBench/data.zip`.

2. **Run Evaluation**:

   ```bash
   bash scripts/test_longbench.sh
   ```

3. **Evaluation Results**:

   Below are the evaluation logs for various models and configurations:

   **LLaMA2-7B**

   | Method        | Config       | Eval Log                                                                                                           |
   |---------------|--------------|--------------------------------------------------------------------------------------------------------------------|
   | Original      | N/A          | [llama2-7b](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b.json)                     |
   | +Quest        | 1024 Budget  | [llama2-7b-quest-1024](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-quest-1024.json) |
   | +Quest        | 128 Budget   | [llama2-7b-quest-128](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-quest-128.json)   |
   | +MagicPIG     | Default      | [llama2-7b-magicpig](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-magicpig.json)     |
   | +Spotlight    | 90% Pruned   | [llama2-7b-spotlight-90](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-spotlight-90.json) |
   | +Spotlight    | 98% Pruned   | [llama2-7b-spotlight-98](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-spotlight-98.json) |

   **LLaMA2-7B-Chat**

   | Method        | Config       | Eval Log                                                                                                                     |
   |---------------|--------------|------------------------------------------------------------------------------------------------------------------------------|
   | Original      | N/A          | [llama2-7b-chat](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-chat.json)                     |
   | +Quest        | 1024 Budget  | [llama2-7b-chat-quest-1024](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-chat-quest-1024.json) |
   | +Quest        | 128 Budget   | [llama2-7b-chat-quest-128](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-chat-quest-128.json)   |
   | +MagicPIG     | Default      | [llama2-7b-chat-magicpig](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-chat-magicpig.json)     |
   | +Spotlight    | 90% Pruned   | [llama2-7b-chat-spotlight-90](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-chat-spotlight-90.json) |
   | +Spotlight    | 98% Pruned   | [llama2-7b-chat-spotlight-98](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama2-7b-chat-spotlight-98.json) |

   **LLaMA3-8B**

   | Method        | Config       | Eval Log                                                                                                           |
   |---------------|--------------|--------------------------------------------------------------------------------------------------------------------|
   | Original      | N/A          | [llama3-8b](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama3-8b.json)                     |
   | +Quest        | 1024 Budget  | [llama3-8b-quest-1024](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama3-8b-quest-1024.json) |
   | +Quest        | 256 Budget   | [llama3-8b-quest-256](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama3-8b-quest-256.json)   |
   | +MagicPIG     | Default      | [llama3-8b-magicpig](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama3-8b-magicpig.json)     |
   | +Spotlight    | 90% Pruned   | [llama3-8b-spotlight-90](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama3-8b-spotlight-90.json) |
   | +Spotlight    | 98% Pruned   | [llama3-8b-spotlight-98](https://anonymous.4open.science/r/spotlight/test_longbench/log/llama3-8b-spotlight-98.json) |

### <span id="eval-fidelity"> QA Response Fidelity </span>

To evaluate response fidelity, obtain LongBench output files by either:

- Running the LongBench test scripts to generate output files.
- Using provided output files from the [LongBench](#eval-longbench) section.

For example, to compare output similarity between LLaMA3-8B with and without Spotlight Attention:

```bash
python test_longbench/test_sim.py test_longbench/log/llama3-8b.json test_longbench/log/llama3-8b-spotlight.json
```

---

## <span id="training"> 🔥 Training </span>

### 1. Create Directories

```bash
cd spotlight
mkdir -p data/slimpajama ckp
```

### 2. Download Datasets

Download and place the following datasets in the `data/slimpajama` directory:
- [arxiv.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/slimpajama/arxiv.json)
- [book.json](https://huggingface.co/datasets/namespace-Pt/long-llm-data/blob/main/slimpajama/book.json)

### 3. Training Execution

Choose a training method based on available disk space:

- **Sufficient Disk Space**:
  1. Edit `train.sh` to include the `--prepare_data` argument and run the script.
  2. After completion, replace `--prepare_data` with `--use_prepared_data` and rerun.

- **Limited Disk Space** (Default):
  Generate activations on-the-fly by running the `train.sh` script without modifications.

The trained checkpoint will be saved in the `ckp` directory and can be referenced in test scripts using the `load_ckp` keyword.

### 4. Memory Optimization

Training involves computing a ranking loss, which can be memory-intensive due to the large tensor \( Z \):

\[
n_{\text{heads}} \times n_{\text{query}} \times n_{\text{top}} \times (n_{\text{query}} - n_{\text{top}})
\]

To mitigate memory issues, adjust the following parameters (default: 1024):
- `--max_que`: Maximum query tokens.
- `--max_top`: Maximum top-ranked tokens.
- `--max_oth`: Maximum other tokens.

If out-of-memory errors occur, reduce `--max_que` and `--max_oth` first.

---

This README provides a polished, professional guide to installing, evaluating, and training models with Spotlight Attention. For additional details or support, refer to the linked repositories or datasets.