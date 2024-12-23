import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }

def preprocess_dataset():
    from datasets import load_dataset
    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Split dataset and tokenize
    train_texts = dataset["train"]["text"][:5000]
    train_labels = dataset["train"]["label"][:5000]
    test_texts = dataset["test"]["text"][:1000]
    test_labels = dataset["test"]["label"][:1000]

    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    test_dataset = SentimentDataset(test_texts, test_labels, tokenizer)
    return train_dataset, test_dataset, tokenizer
