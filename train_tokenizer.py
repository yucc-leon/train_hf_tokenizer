import os, json
from datasets import load_dataset
from tqdm import tqdm
from tokenizers import (
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
    Tokenizer,
)


os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
# os.environ['HF_HOME'] = "/home/aigc/.cache/huggingface"

corpus_for_tokenizer = load_dataset("./datasets/to_train_tokenizer_sample")


print("corpus info:")
print(corpus_for_tokenizer)


def get_training_corpus():
    for i in range(0, 10000, 1000):
        yield corpus_for_tokenizer['train'][i : i+1000]['text']


special_tokens = [
    "<｜begin▁of▁text｜>",
    "<｜end▁of▁text｜>",
    "<｜User｜>", 
    "<｜Assistant｜>", 
    "<｜fim▁middle｜>",
    "<｜fim▁prefix｜>",
    "<｜fim▁suffix｜>",
    "<｜api▁calls▁begin｜>",
    "<｜api▁calls▁end｜>",
    "<｜api▁call▁begin｜>",
    "<｜api▁call▁end｜>",
    "<｜api▁outputs▁begin｜>",
    "<｜api▁outputs▁end｜>",
    "<｜api▁output▁begin｜>",
    "<｜api▁output▁end｜>",
    "<｜api▁sep｜>",
]        



tokenizer = Tokenizer(models.BPE())


tokenizer.normalizer = normalizers.Sequence([])

# preprocessing copied from deepseekV2
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Split('[\r\n]', "isolated"),   
        pre_tokenizers.Split("\s?[A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃]+","isolated"), 
        pre_tokenizers.Split('\s?[!-/:-~！-／：-～‘-‟　-。]+', "isolated"),   
        pre_tokenizers.Split('\s+$', "isolated"), 
        pre_tokenizers.Split("[一-龥ࠀ-一가-퟿]+","isolated"),
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
tokenizer.post_processor = processors.ByteLevel(use_regex=True, trim_offsets=False, add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel(use_regex=True, trim_offsets=True, add_prefix_space=True)



print("start training...")
trainer = trainers.BpeTrainer(vocab_size=1024, special_tokens=special_tokens)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)


from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<｜begin▁of▁text｜>",
    eos_token="<｜end▁of▁text｜>",
    pad_token="<｜end▁of▁text｜>"
)

wrapped_tokenizer.save_pretrained("./64k_tokenizer")