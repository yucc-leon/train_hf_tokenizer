This is an experimental repository for nearly everything you fundamentally need to know about:
# Training Your Customized Tokenizer
feel free to check and modify these files.

## Updated: lesson learned
- This tokenizer was initially used to train a relatively small decoder-only model, which appeared promising based on the metrics. However, after several training steps, the channel loss curves behaved abnormally. They continued to decrease on the Chinese and code datasets but plateaued quickly on the English datasets.
- We conducted ablations across architectures, optimizers, hyperparameters, and tokenizers, and this tokenizer emerged as the most suspicious. Upon switching to other tokenizers, such as those from Neo, GPT-NeoX, LlaMA, and others, the channel loss curves showed more balanced downward trends across different datasets.
- In an attempt to diagnose the issue, I compared this tokenizer with higher-quality ones like DeepSeek’s. For the first few thousand (sub/byte-level) tokens, they appeared quite similar; for the latter cases, my tokenizer contained many abnormal tokens such as `///Ġ<summary>ĊĠĠĠĠĠĠĠĠ`, `form-control`, `one.Ċ`, and `}ĊĠĠĠĠĠĠĠĠ}ĊĠĠĠĠĠĠĠĠ`. These likely stemmed from the unclean non-code corpus and the code-related corpus.
- Here’s the key lesson: When should you train your own tokenizer, and how should you approach it?
  - Train your own tokenizer if you have special use cases that differ significantly from “universal” models like GPTs, LLaMAs, or Qwens. Domain-specific tokens can represent domain knowledge, potentially accelerating inference speed. 
  - However, be aware that the training corpus must be meticulously curated. People often lack deep insights into the corpus they possess, and Byte-Pair Encoding (BBPE) can make it challenging to inspect and understand the data. Your corpus for training tokenizers should be both very clean and sufficiently large to accommodate tens of thousands of tokens. 
  - Despite these challenges, training a tokenizer using modern tools isn’t overly difficult.  

## Basic Infomation
Inspired by DeepSeek-LLM’s tech report, which mentioned that their researchers trained tokenizers only using Huggingface’s Tokenizer library, I decided to train my own as well.

- datasets
  - m-a-p/Matrix: A large dataset containing code, Chinese, and English corpora
  - skypile: a high-quality Chinese corpus for pretraining
- results

| model | code | en | zh/bi |
|---|---|---|---|
| Meta-Llama-3-8B | 4.29 | 4.63 | 3.57 |
| Yi-1.5-9B-Chat | 3.29 | 4.25 | 4.24 |
| CodeQwen1.5-7B | 3.33 | 4.26 | 4.63 |
| DeepSeek-Coder-V2-Lite-Instruct | 3.56 | 4.37 | 4.14 |
| neo_7b_sft_v0.1 | 2.25 | 4.18 | 4.26 |
| mine | 5.32 | 5.51 | 4.71 |

Tested using sampled datasets:
- code search net (code)
- fineweb (en)
- skypile testset (zh)

These compression rates are unexpectedly high, given my tiny training corpus and the simplicity of my implementation, which lacked any optimization. Larger training sets and different test sets could produce significantly different results.


## Tips

1. train_tokenizer.py is a basic example of using a processed corpus to train your tokenizer. The settings are largely borrowed from DeepSeek-LLM V2. Their preprocessing steps are complex, but I suspect they are effective for handling web data.
2. parallel_*.py scripts are for processing the corpus in parallel.
3. You can use train_corpus_stats.py to calculate the number of tokens (using your tokenizer) trained after completing the training. Adding domain-specific information can be useful in practice.
4. You can compute the compression rate using compression_rate.py.
5. I’ve removed some private paths, so you’ll need to update them with your own to make everything work.
6. Training tokenizers is both memory and time-intensive. In my experiments, 1.5 TB of memory wasn’t sufficient for a 27 GB corpus; I downsampled it to 18 GB (about 3B tokens) and ran it on a 1.3 TB memory server, which took about 5 hours. For comparison, the m-a-p/Neo tokenizer was trained on 50B tokens over 12 days, with an unspecified amount of memory, according to its author.
7. My tokenizer consumed less computation, so I don’t believe it outperforms the other models listed above.


## References

I really learned something from
1. Karpathy's video lecture (https://www.youtube.com/watch?v=zduSFxRajkE) for:
 - you can use https://tiktokenizer.vercel.app to visualize different tokenizing methods using several tokenizers
 - how does BPE work and a simple way to implement it
 - GPT-2/GPT-4 tokenizers, special tokens, sentencepiece, vocab configs, multimodal tokenizers
 - ...

2. Huggingface's NLP course (https://huggingface.co/learn/nlp-course/chapter6/1). You can train your tokenizer by their examples step by step.
3. DeepSeek-LLM V2's tokenizer
4. Other references
 - https://github.com/huggingface/transformers/issues/4777
 - https://github.com/facebookresearch/fairseq/issues/1716
 - https://github.com/openai/gpt-2/issues/80
 - https://github.com/huggingface/transformers/issues/1083#issuecomment-524303077
 - https://github.com/huggingface/tokenizers/issues/1218
 - https://guillaume-be.github.io/2021-09-16/byte_pair_encoding
 - https://guillaume-be.github.io/2021-09-16/byte_pair_encoding
 - https://juejin.cn/post/7088322473640329230 (in Chinese)
 - https://mingchao.wang/ZfZEmye9/ (in Chinese)
