import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


from transformers import PreTrainedTokenizerFast 
from datasets import load_dataset



fw_small_set = load_dataset("/home/test/.cache/huggingface/datasets/HuggingFaceFW___fineweb", name="sample-10BT", split="train", streaming=False)
skypile_testset = load_dataset("/home/test/.cache/huggingface/datasets/skywork___chinese_domain_modeling_eval", split='train')


code_compression_test = codesearchnet_dataset['train'].shuffle(seed=42).select(range(100000))
# whole_func_string
en_compression_test   = fw_small_set.shuffle(seed=42).select(range(100000))
# text
skypile_compression_test = skypile_testset
# text


from tqdm import tqdm



test_corpus = [code_compression_test, en_compression_test, skypile_compression_test]
corpus_cate = ['code', 'en', 'zh/bi']
text_fields = ['whole_func_string', 'text', 'text']
byte_count = dict()

tokenizers = [
    'm-a-p/neo_7b_sft_v0.1',
    'meta-llama/Meta-Llama-3-8B',
    '01-ai/Yi-1.5-9B-Chat', 
    'Qwen/CodeQwen1.5-7B', 
    'deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct',
    "./64k_tokenizer"
]



from copy import deepcopy

results_template = {
    'cate': [],
    'byte_len': [],
    'token_len': [],
    'compression_rate': []
}
result_aggr = dict()
for tokenizer_path in tokenizers:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer_name = tokenizer_path.split("/")[-1]
    print(f"testing {tokenizer_name}...")
    result = deepcopy(results_template)
    for cate, corpus, text_field in zip(corpus_cate, test_corpus, text_fields):
        char_count, token_count = 0, 0
        result['cate'].append(cate)
        for data in tqdm(corpus):
            text = data[text_field]
            char_count += len(text.encode("utf-8"))
            token_count += len(tokenizer.tokenize(text))
        result['byte_len'].append(char_count)
        result['token_len'].append(token_count)
        result['compression_rate'].append(char_count/token_count)
        
    result_aggr[tokenizer_name] = result


print(result_aggr)

# copy this dict into your Claude.ai (recommended) or ChatGPT (no artifact with free account) 
# and let them transform it into a good looking table