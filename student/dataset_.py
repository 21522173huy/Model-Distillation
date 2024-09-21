
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import DataCollatorWithPadding

class SentimentDataset(Dataset):
    def __init__(self, split, tokenizer, subset_ratio=1.0):
        self.dataset = load_dataset('tweet_eval', 'sentiment')[split]
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Take the first subset_ratio% of the dataset
        subset_size = int(len(self.dataset) * subset_ratio)
        self.dataset = self.dataset.select(range(subset_size))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item['text']
        label = item['label']
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=256)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(label)
        return inputs

def create_dataloaders(tokenizer, batch_size, subset_ratio=0.15):
    train_dataset = SentimentDataset(split='train', tokenizer=tokenizer, subset_ratio=subset_ratio)
    test_dataset = SentimentDataset(split='test', tokenizer=tokenizer, subset_ratio=subset_ratio)
    val_dataset = SentimentDataset(split='validation', tokenizer=tokenizer, subset_ratio=subset_ratio)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.data_collator
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_dataset.data_collator
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.data_collator
    )

    return train_dataloader, test_dataloader, val_dataloader
