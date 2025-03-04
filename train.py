import torch
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader


def preprocess_function(examples):
    de_texts = [translation['de'] for translation in examples['translation']]
    en_texts = [translation['en'] for translation in examples['translation']]
    
    de_inputs = de_tokenizer(de_texts, truncation=True, max_length=512, padding="max_length")
    en_inputs = en_tokenizer(en_texts, truncation=True, max_length=512, padding="max_length")
    
    model_inputs = {
        "input_ids": de_inputs["input_ids"],
        "attention_mask": de_inputs["attention_mask"],
        "decoder_input_ids": en_inputs["input_ids"],
        "decoder_attention_mask": en_inputs["attention_mask"],
        "labels": en_inputs["input_ids"].copy()
    }
    
    return model_inputs

wmt14 =  load_dataset("wmt/wmt14", "de-en")
train_subset = wmt14['train'].select(range(10000))
de_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_dataset = train_subset.map(preprocess_function, batched=True, batch_size=1000)
tokenized_dataset.set_format("torch")
data_collator = DataCollatorWithPadding(tokenizer=en_tokenizer)
train_dataloader = DataLoader(
    tokenized_dataset, shuffle=True, batch_size=8, collate_fn=data_collator
)