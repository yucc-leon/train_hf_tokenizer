This is a experimental notes for almost everything (fundamentally) you need to know about
# Training Your Customized Tokenizer
feel free to check and modify these files.

## Basic Infomation
Inspired by DeepSeek-LLM's tech report, which tells that their researchers train their own tokenizers just using Huggingface's Tokenizer library, I trained my own tokenizers too.

- datasets
  - m-a-p/Matrix, a very big dataset containing code, Chinese, and English corpus
  - skypile, a high quality Chinese corpus for pretraining
- results

| model | code | en | zh/bi |
|---|---|---|---|
| Meta-Llama-3-8B | 4.29 | 4.63 | 3.57 |
| Yi-1.5-9B-Chat | 3.29 | 4.25 | 4.24 |
| CodeQwen1.5-7B | 3.33 | 4.26 | 4.63 |
| DeepSeek-Coder-V2-Lite-Instruct | 3.56 | 4.37 | 4.14 |
| neo_7b_sft_v0.1 | 2.25 | 4.18 | 4.26 |
| mine | 5.32 | 5.51 | 4.71 |

test using sampled datasets:
- code search net (code)
- fineweb (en)
- skypile testset (zh)

These compression rates are too good for very my small training corpus and simple implementation without any optimization. I guess far larger training sets and different test sets would make very different numbers.


## Lectures and Discussions

I really learned something from 
1. Karpathy's video lecture (https://www.youtube.com/watch?v=zduSFxRajkE) and here's what I learned:
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