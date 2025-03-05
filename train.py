import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator
from torch.utils.data import DataLoader
from transformer.transformer import Transformer

GPU_ID = 1
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
# Select a value ]0.0, 1.0] to train on a subset of the train set
TRAIN_PERCENTAGE = 0.01

EMB_DIM = 512
FORWARD_DIM = 4 * EMB_DIM
NUM_HEADS = 8
NUM_LAYERS = 6
MAX_SEQ_LEN = 128
DROPOUT_RATE = 0.1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(SCRIPT_DIR, "out")

def preprocess_function(examples, src_tokenizer, tgt_tokenizer):
    de_texts = [translation['de'] for translation in examples['translation']]
    en_texts = [translation['en'] for translation in examples['translation']]
    
    de_inputs = src_tokenizer(de_texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    en_inputs = tgt_tokenizer(en_texts, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
    
    model_inputs = {
        "input_ids": de_inputs["input_ids"],
        "attention_mask": de_inputs["attention_mask"],
        "decoder_input_ids": en_inputs["input_ids"],
        "decoder_attention_mask": en_inputs["attention_mask"],
        "labels": en_inputs["input_ids"].copy()
    }
    
    return model_inputs

def train_loop(model, tgt_tokenizer, train_dataloader, device, epochs = 1):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.pad_token_id)
    for epoch in range(epochs):
        for i, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
            optimizer.zero_grad()
            
            # move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Add BOS token for the decoder input
            bos_tokens = torch.full((batch['decoder_input_ids'].shape[0], 1), 
                                    tgt_tokenizer.bos_token_id, 
                                    device=device)
            decoder_in = torch.cat([bos_tokens, batch['decoder_input_ids']], dim=1)
            
            # Select up to second-last token for teacher forcing
            decoder_in = decoder_in[:, :-1]

            # Forward pass - shape: batch_size x seq_length x vocab_size
            logits = model(batch["input_ids"], decoder_in)

            # Reshape for loss calculation. Loss_fn expects batch to be flattened
            _, _, vocab_size = logits.shape
            logits_reshaped = logits.contiguous().view(-1, vocab_size)
            labels_reshaped = batch['labels'].contiguous().view(-1)

            loss = loss_fn(logits_reshaped, labels_reshaped)
            loss.backward()
            optimizer.step()
            
            # TODO: log loss properly (to wandb)
            # TODO: log perplexity
            # TODO: evaluate on val set every n steps
            # TODO: save checkpoints
            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wmt14 =  load_dataset("wmt/wmt14", "de-en")
    if TRAIN_PERCENTAGE <= 0.0 or TRAIN_PERCENTAGE > 1.0:
        raise ValueError(f"Train percentage must be ]0.0, 1.0]. Was: {TRAIN_PERCENTAGE}")

    if TRAIN_PERCENTAGE == 1.0:
        train_subset = wmt14['train']
    else:
        train_subset = wmt14['train'].train_test_split(train_size=TRAIN_PERCENTAGE, seed=42)['train']

    print(f"Train set size: {len(train_subset)}")

    de_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # add special tokens to target tokenizer for decoding
    special_tokens_dict = {'bos_token': '<s>', 'eos_token': '</s>'}
    en_tokenizer.add_special_tokens(special_tokens_dict)

    tokenized_dataset = train_subset.map(
        preprocess_function,
        batched=True,
        batch_size=1000,
        fn_kwargs={
        "src_tokenizer": de_tokenizer,
        "tgt_tokenizer": en_tokenizer
        },
        desc="Tokenizing and preprocessing dataset"
    )
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"])

    data_collator = DefaultDataCollator()
    train_dataloader = DataLoader(
        tokenized_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
    )

    model = Transformer(
        len(de_tokenizer),
        len(en_tokenizer),
        de_tokenizer.pad_token_id,
        en_tokenizer.pad_token_id,
        forward_dim=FORWARD_DIM,
        emb_dim=EMB_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_SEQ_LEN,
        dropout_rate=DROPOUT_RATE
    )
    model.to(device)

    train_loop(model, en_tokenizer, train_dataloader, device)

    # TODO: also save model architecture
    os.makedirs(OUT_PATH, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT_PATH, "state_dict.pt"))

if __name__ == "__main__":
    main()