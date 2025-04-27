"""
Custom text dataset loader for sarcasm detection using PyTorch.
Inspired by Aladdin Persson's 

Modified by [Your Name]
2025-04-15
"""

import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from preprocessing import preprocess_text

from read_integrate_all_data_sources import read_all_data_sources
from sklearn.model_selection import train_test_split
# Load spaCy tokenizer
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, text_list):
        frequencies = {}
        idx = 4

        for sentence in text_list:
            for word in self.tokenizer(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized
        ]


class SarcasmDataset(Dataset):
    def __init__(self, dataframe, vocab, preprocessing_fn=preprocess_text, preprocessing_args=None):
        self.texts = dataframe["text"].fillna("").tolist()
        self.labels = dataframe["sarcastic"].tolist()
        self.vocab = vocab
        self.preprocessing_fn = preprocessing_fn
        self.preprocessing_args = preprocessing_args if preprocessing_args else {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)

        # Apply preprocessing here if function is provided
        if self.preprocessing_fn:
            text = self.preprocessing_fn(text, **self.preprocessing_args)

        numericalized = [self.vocab.stoi["<SOS>"]]
        numericalized += self.vocab.numericalize(text)
        numericalized.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized), label


class TextCollate:
    def __init__(self, pad_idx, max_len=50):
        self.pad_idx = pad_idx
        self.max_len = max_len

    def __call__(self, batch):
        texts = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch])
        truncated = [seq[:self.max_len] for seq in texts]
        padded = pad_sequence(truncated, batch_first=True,
                              padding_value=self.pad_idx)
        return padded, labels


def get_text_loader(dataframe, freq_threshold=5, batch_size=32, shuffle=True):
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(dataframe["text"].fillna("").tolist())

    dataset = SarcasmDataset(dataframe, vocab)
    pad_idx = vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=TextCollate(pad_idx),
    )

    return loader, vocab


_, _, _, _, _, all_data = read_all_data_sources()
all_data['text'] = all_data['text'].astype(str)
all_data['sarcastic'] = all_data['sarcastic'].astype(int)


train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42)

train_loader, vocab = get_text_loader(train_df, batch_size=64)
test_loader, _ = get_text_loader(test_df, batch_size=64)

for batch in train_loader:
    x, y = batch
    print(x.shape)  # e.g. torch.Size([64, seq_len])
    print(y.shape)  # e.g. torch.Size([64])
    break
