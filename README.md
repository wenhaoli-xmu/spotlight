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


## <span id="eval-nlu"> Few-Shot Learning </span>


## <span id="eval-longbench"> LongBench </span>

## <span id="eval-fidelity"> Output Fidelity </span>


## <span id="eval-needle"> Needle In A Haystack </span>