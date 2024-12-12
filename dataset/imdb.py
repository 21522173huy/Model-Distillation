
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import DataCollatorWithPadding
import pandas as pd

class IMDB_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, rdrsegmenter=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def __len__(self):
        return len(self.dataset)

    def remove_special_tokens(self, text):
        special_tokens = ['br />', '#', '$', '%', '&', '*', '+', '-', '/', '<', '=', '>', '?', '@', '^', '_', '`', '~', '\\']
        for token in special_tokens:
            text = text.replace(token, '')
        return text

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.rdrsegmenter is None:
            text = item['text']
        else:
            output = self.rdrsegmenter.word_segment(item['text'])
            text = ' '.join(output)
        label = item['label']
        text = self.remove_special_tokens(text)
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=256)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(label)
        return inputs

def create_dataloaders(tokenizer, batch_size, full_test = False):
    # Load dataset
    sentiment_dataset = load_dataset('stanfordnlp/imdb')

    # Train, Val, Test
    train_val_split = sentiment_dataset['train'].train_test_split(test_size=0.2)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']
    
    if full_test == True: test_dataset = sentiment_dataset['test']
    else : 
        test_split_dataset = sentiment_dataset['test'].train_test_split(test_size=0.2)
        test_dataset = test_split_dataset['test']

    # Modify Dataset
    train_dataset = IMDB_Dataset(dataset=train_dataset, tokenizer=tokenizer)
    val_dataset = IMDB_Dataset(dataset=val_dataset, tokenizer=tokenizer)
    test_dataset = IMDB_Dataset(dataset=test_dataset, tokenizer=tokenizer)

    print(f'Train Length: {len(train_dataset)}')
    print(f'Validation Length: {len(val_dataset)}')
    print(f'Test Length: {len(test_dataset)}')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.data_collator
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.data_collator
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.data_collator
    )

    return train_dataloader, val_dataloader, test_dataloader
