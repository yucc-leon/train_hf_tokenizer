from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

import os
import random
import multiprocessing

from multiprocessing import Pool, cpu_count


tokenizer = AutoTokenizer.from_pretrained("./64k_tokenizer")

corpus = load_dataset("./datasets/to_train_tokenizer_sample")

print("corpus info:")
print(corpus)

token_counter = 0

def process_chunks(chunk):
    """
    Process a list of chunks by selecting a fraction of lines randomly and writing them to the output file.

    Args:
        chunks (list): List of chunks to process.
        sample_fraction (float): Fraction of lines to keep (between 0 and 1).
        output_file (str): Path to the output file where the sampled data will be saved.
        tokenizer (transformers.tokenization_utils_base.PreTrainedTokenizer): Tokenizer to use.
    """
    
    return len(tokenizer.tokenize(chunk['text']))
    

result = 0
chunk_size = 32000
chunks = []
with Pool(16) as pool:
    for line in tqdm(corpus['train']):
        chunks.append(line)
        if len(chunks) == chunk_size:
            results = pool.map(process_chunks, chunks)
        
            result += sum(results)
            chunks = []
    if chunks:
        results = pool.map(process_chunks, chunks)
        result += sum(results)

print(f"train corpus token count: {result:,}")

