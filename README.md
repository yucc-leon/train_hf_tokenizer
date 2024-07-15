This is a experimental repo for almost everything (fundamentally) you need to know about
# Training Your Customized Tokenizer
feel free to check and modify these files.

## Basic Infomation
Inspired by DeepSeek-LLM's tech report, which says that their researchers train their tokenizers just using Huggingface's Tokenizer library, I trained my tokenizers too.

- datasets
  - m-a-p/Matrix, a very big dataset containing code, Chinese, and English corpus
  - skypile, a high-quality Chinese corpus for pretraining
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

These compression rates are too good for my tiny training corpus and simple implementation without any optimization. Far larger training sets and different test sets may make very different numbers.

## Tips

1. train_tokenizer.py is a simple example using a processed corpus to train your tokenizer. Basically copied the configs used in DeepSeek-LLM V2. Their preprocessing is abstruse but I guess it's useful to deal with web data.
2. parallel_*.py-s are processing the corpus in parallel. 
3. you can use train_corpus_stats.py to compute how many tokens (using your tokenizer) you've trained after finishing training. Adding domain info would be useful in practice.
4. you can compute compression rate using compression_rate.py
5. I'd removed some private paths so you need to change them to yours to make them work
6. training tokenizers is very memory-and-time-consuming. In my experiments, 1.5T is not enough for 27G's corpus; I downsampled it to 18G and ran it on a 1.3T memory server host for about 5 hours. In comparison, m-a-p/Neo's tokenizer was trained on 50B tokens with 12 days and unknown-size memory, said by its author.
7. mine took less computation so I don't think it's better than the other players in the above list.


## References

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
