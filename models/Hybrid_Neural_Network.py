
from itertools import product
import os
import json
import random
import numpy as np
import torch
from transformers import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ====================
# Data Loading & Preprocessing
# ====================
PROJECT_ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../."))
DATA_DIR = os.path.join(PROJECT_ROOT_DIR, "data")
DATA_FILE = os.path.join(
    DATA_DIR, "raw", "News Headlines", "Sarcasm_Headlines_Dataset_v2.json")
glove_path = os.path.join(
    DATA_DIR, "embeddings", "glove.6B.300d.txt")

best_model_path = os.path.join(
    PROJECT_ROOT_DIR, "results", "lstm", "best_model_last.pth")
# Parse the JSON data (each line is a JSON record)
data = []
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

# Extract headlines and labels
headlines = [item['headline'] for item in data]
labels = [item['is_sarcastic'] for item in data]


# Text preprocessing
def tokenize(text):
    # Tokenize input text
    tokens = tokenizer.tokenize(text)
    return tokens[:25]  # Limit to 25 tokens for consistency


tokenized_headlines = [tokenize(h) for h in headlines]

# Split data into train/validation/test (80/10/10)
train_sentences, temp_sentences, train_labels, temp_labels = train_test_split(
    tokenized_headlines, labels, test_size=0.2, random_state=42, stratify=labels
)
val_sentences, test_sentences, val_labels, test_labels = train_test_split(
    temp_sentences, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# Build vocabulary from training data
vocab = {'<PAD>': 0, '<UNK>': 1}
for sentence in train_sentences:
    for word in sentence:
        if word not in vocab:
            vocab[word] = len(vocab)

vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

# Convert sentences to index sequences (with padding)


def encode_and_pad(sentences, vocab, max_len=None):
    sequences = []
    for sent in sentences:
        seq = [vocab.get(w, vocab['<UNK>']) for w in sent]
        sequences.append(seq)
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)
    # Pad sequences to max_len
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [vocab['<PAD>']] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])
    return np.array(padded), max_len


# Encode and pad all splits
train_sequences, max_len = encode_and_pad(train_sentences, vocab)
val_sequences, _ = encode_and_pad(val_sentences, vocab, max_len)
test_sequences, _ = encode_and_pad(test_sentences, vocab, max_len)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)
test_labels = np.array(test_labels)

print(f"Max sequence length: {max_len}")

# ====================
# Prepare Embeddings (Word2Vec)
# ====================
embedding_dim = 300  # dimension for embeddings

# Initialize embedding matrix with random for all words (will overwrite known words)
embedding_matrix = np.random.uniform(-0.25, 0.25,
                                     (vocab_size, embedding_dim)).astype(np.float32)

# used GloVe embeddings(300d)
embedding_index = {}
with open(glove_path, 'r', encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Assign pretrained embeddings to our matrix
for word, idx in vocab.items():
    if word in embedding_index:
        embedding_matrix[idx] = embedding_index[word]
# Convert embedding matrix to tensor
embedding_matrix = torch.tensor(embedding_matrix)


# ====================
# Dataset and DataLoader
# ====================


class SarcasmDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


batch_size = 64
train_dataset = SarcasmDataset(train_sequences, train_labels)
val_dataset = SarcasmDataset(val_sequences, val_labels)
test_dataset = SarcasmDataset(test_sequences, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# ====================
# Model Definition: Hybrid CNN + BiLSTM + Attention
# ====================


class SarcasmModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, embedding_weights, num_filters=100, filter_sizes=[3, 4, 5],
                 lstm_hidden=128, fc_hidden=100, dropout_prob=0.5, lstm_layers=1):
        super(SarcasmModel, self).__init__()
        # Embedding layer, initialize with pre-trained or random embeddings
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=vocab['<PAD>'])
        self.embedding.weight.data.copy_(embedding_weights)
        self.embedding.weight.requires_grad = True  # fine-tune embeddings

        # Convolution layers (CNN for n-gram features)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters,
                      kernel_size=fs, padding=fs//2)
            for fs in filter_sizes
        ])

        # BiLSTM layer
        self.bilstm = nn.LSTM(embed_dim, lstm_hidden, bidirectional=True,
                              batch_first=True, num_layers=lstm_layers)

        # Attention mechanism (Bahdanau-style) parameters
        self.attn_linear = nn.Linear(lstm_hidden*2, lstm_hidden*2)
        self.attn_vector = nn.Parameter(torch.randn(lstm_hidden*2))

        # Fully connected layers for final classification
        total_features = num_filters * len(filter_sizes) + (lstm_hidden*2)
        self.fc = nn.Linear(total_features, fc_hidden)
        self.out = nn.Linear(fc_hidden, 2)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        emb = emb.permute(0, 2, 1)  # [batch, embed_dim, seq_len] for conv

        # CNN branch: apply each convolution + relu + maxpool
        conv_outputs = []
        for conv in self.convs:
            c = conv(emb)          # [batch, num_filters, seq_len]
            c = F.relu(c)
            # Global max pooling over time dimension
            c = F.max_pool1d(c, kernel_size=c.shape[2]).squeeze(
                2)  # [batch, num_filters]
            conv_outputs.append(c)
        # [batch, num_filters * len(filter_sizes)]
        cnn_feat = torch.cat(conv_outputs, dim=1)

        # BiLSTM branch: use original embedding
        emb2 = self.embedding(x)  # [batch, seq_len, embed_dim]
        lstm_out, _ = self.bilstm(emb2)  # [batch, seq_len, lstm_hidden*2]

        # Attention: calculate weights and context vector
        # [batch, seq_len, hidden*2]
        u = torch.tanh(self.attn_linear(lstm_out))
        # [batch, seq_len, hidden*2] dot [hidden*2] -> [batch, seq_len]
        scores = torch.matmul(u, self.attn_vector)
        attn_weights = F.softmax(scores, dim=1)  # [batch, seq_len]
        # Compute weighted sum of LSTM outputs (context vector)
        context = torch.bmm(attn_weights.unsqueeze(
            1), lstm_out).squeeze(1)  # [batch, hidden*2]

        # Combine CNN and attention features
        # [batch, total_features]
        combined = torch.cat([cnn_feat, context], dim=1)
        combined = self.dropout(combined)
        hidden = F.relu(self.fc(combined))
        hidden = self.dropout(hidden)
        output = self.out(hidden)  # [batch, 2]
        return output, attn_weights


# ====================
# Training & Evaluation
# ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


best_model = None

# Define hyperparameter search space
param_grid = {
    'lr': [1e-3, 5e-4, 5e-5, 5e-6],
    'dropout_prob': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'lstm_layers': [1, 2],
    'num_filters': [100, 150],
    'filter_sizes': [[3, 4, 5], [2, 3, 4], [1, 3, 5]],
    'lstm_hidden': [64, 128],
    'fc_hidden': [50, 100, 128]
}

# Generate all combinations
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in product(*values)]
print(f"Total combinations: {len(param_combinations)}")

best_val_acc = 0
best_model = None
best_params = {}

for params in param_combinations:
    print(f"\nTesting params: {params}")

    # Initialize model with current params
    model = SarcasmModel(
        vocab_size,
        embedding_dim,
        embedding_matrix,
        num_filters=params['num_filters'],
        filter_sizes=params['filter_sizes'],
        lstm_hidden=params['lstm_hidden'],
        fc_hidden=params['fc_hidden'],
        dropout_prob=params['dropout_prob'],
        lstm_layers=params['lstm_layers']
    ).to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    # Training loop (shortened for grid search)
    for epoch in range(5):  # Fewer epochs for faster search
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = model(batch_x)
                _, preds = torch.max(outputs, 1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            best_params = params.copy()

    print(f"Val Acc: {val_acc:.4f}")

# Save best model
torch.save(best_model, best_model_path)
print(f"\nBest Params: {best_params}")

# Evaluate on test set using best model
model = SarcasmModel(
    vocab_size,
    embedding_dim,
    embedding_matrix,
    num_filters=params['num_filters'],
    filter_sizes=params['filter_sizes'],
    lstm_hidden=params['lstm_hidden'],
    fc_hidden=params['fc_hidden'],
    dropout_prob=params['dropout_prob'],
    lstm_layers=params['lstm_layers']
).to(device)

model.load_state_dict(torch.load(r'best_model_last.pth'))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs, _ = model(batch_x)
        _, preds = torch.max(outputs, 1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")
