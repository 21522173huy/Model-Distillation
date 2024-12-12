
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Dataset as HFDataset
from transformers import DataCollatorWithPadding
from imblearn.over_sampling import RandomOverSampler
import pandas as pd

def apply_random_oversampling(dataset):
    df = pd.DataFrame(dataset)
    X = df['sentence']
    y = df['sentiment']

    ros = RandomOverSampler()
    X_resampled, y_resampled = ros.fit_resample(X.values.reshape(-1, 1), y)

    df_resampled = pd.DataFrame({'sentence': X_resampled.flatten(), 'sentiment': y_resampled})
    return HFDataset.from_pandas(df_resampled)

class UitVSFC_Dataset(Dataset):
    def __init__(self, dataset, tokenizer, rdrsegmenter=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.rdrsegmenter = rdrsegmenter
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.rdrsegmenter is None:
            text = item['sentence']
        else:
            output = self.rdrsegmenter.word_segment(item['sentence'])
            text = ' '.join(output)
        label = item['sentiment']
        inputs = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt", max_length=256)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = self.swap_tensor_values(torch.tensor(label))
        return inputs

    def swap_tensor_values(self, input_tensor):
        output_tensor = input_tensor.clone()
        output_tensor[output_tensor == 1] = 3  # Temporarily change 1 to 3
        output_tensor[output_tensor == 2] = 1  # Change 2 to 1
        output_tensor[output_tensor == 3] = 2  # Change 3 to 2
        return output_tensor

def create_dataloaders(tokenizer, batch_size, rdrsegmenter = None):
    # Load dataset
    sentiment_dataset = load_dataset("uitnlp/vietnamese_students_feedback")

    # Train, Val, Test
    train_dataset = sentiment_dataset['train']
    train_dataset_resampled = apply_random_oversampling(train_dataset)

    val_dataset = sentiment_dataset['validation']
    val_dataset_resampled = apply_random_oversampling(val_dataset)

    test_dataset = sentiment_dataset['test']

    # Modify Dataset
    train_dataset = UitVSFC_Dataset(dataset=train_dataset_resampled, tokenizer=tokenizer, rdrsegmenter=rdrsegmenter)
    val_dataset = UitVSFC_Dataset(dataset=val_dataset_resampled, tokenizer=tokenizer, rdrsegmenter=rdrsegmenter)
    test_dataset = UitVSFC_Dataset(dataset=test_dataset, tokenizer=tokenizer, rdrsegmenter=rdrsegmenter)

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
