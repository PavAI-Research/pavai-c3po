"""LLM Models and KNOWLEDGE CATALOG"""

LLM_MODEL_KX_CATALOG_TEXT = """

PavAI.Vocie use many AI technologies to implement it's functionality.
Below is a list of large language models has been evaluated. 

### Zephyr 7B 
> Zephyr is a series of language models that are trained to act as helpful assistants. Zephyr-7B-β is the second model in the series, and is a fine-tuned version of mistralai/Mistral-7B-v0.1 that was trained on on a mix of publicly available, synthetic datasets using Direct Preference Optimization (DPO). 

Model description

> Model type: A 7B parameter GPT-like model fine-tuned on a mix of publicly available, synthetic datasets.
> Language(s) (NLP): Primarily English
> License: MIT
> Finetuned from model: [mistralai/Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)            

prompt_template: chatml 

### LlamaGuard-7B 

Llama-Guard is a 7B parameter Llama 2-based input-output safeguard model. It can be used for classifying content in both LLM inputs (prompt classification) and in LLM responses (response classification). It acts as an LLM: it generates text in its output that indicates whether a given prompt or response is safe/unsafe, and if unsafe based on a policy, it also lists the violating subcategories. 

Note: 
You need obtain Meta approval in to the Hugging Face Hub to use the model.


### Functionary-7B-v1
The model to execute function calls

### Mistral-7B-Instruct-v0.2

The Mistral-7B-Instruct-v0.2 Large Language Model (LLM) is an improved instruct fine-tuned version of [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1).

For full details of this model please read our [paper](https://arxiv.org/abs/2310.06825) and [release blog post](https://mistral.ai/news/la-plateforme/).

*** Instruction format ***

In order to leverage instruction fine-tuning, your prompt should be surrounded by `[INST]` and `[/INST]` tokens. The very first instruction should begin with a begin of sentence id. The next instructions should not. The assistant generation will be ended by the end-of-sentence token id.

### Mixtral-8x7B
The Mixtral-8x7B Large Language Model (LLM) is a pretrained generative Sparse Mixture of Experts. The Mixtral-8x7B outperforms Llama 2 70B on most benchmarks we tested.

For full details of this model please read our [release blog post](https://mistral.ai/news/mixtral-of-experts/).


### AdaptLLM / law-LLM-13B
domain-specific models 

### AdaptLLM / medicine-chat
domain-specific models 

### AdaptLLM/finance-LLM-13B
domain-specific models 
            
"""  

