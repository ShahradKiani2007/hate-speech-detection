import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gensim.models import Word2Vec
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight

class TweetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix,
                 hidden_dim=64, num_classes=3, pad_idx=0,
                 bidirectional=True, dropout=0.3, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0  # only applies with num_layers>1
        )
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_out_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)                     # (batch, seq_len, emb_dim)
        lstm_out, (h_n, c_n) = self.lstm(emb)

        if self.lstm.bidirectional:
            # concat final forward + backward hidden states
            h = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch, hidden_dim*2)
        else:
            h = h_n[-1]                               # (batch, hidden_dim)

        out = self.dropout(h)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)                           # logits, no softmax (CrossEntropyLoss does it)
        return out

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels

def encode(text, vocab, max_len=50):
    tokens = text.split()
    ids = [vocab.get(w, vocab["<OOV>"]) for w in tokens][:max_len]
    ids = ids + [vocab["<PAD>"]] * (max_len - len(ids))
    return ids

def build_vocab(texts, max_vocab=20000, min_freq=1):
    counter = Counter()
    for t in texts:
        counter.update(t.split())

    vocab = {"<PAD>": 0, "<OOV>": 1}
    for word, freq in counter.most_common(max_vocab):
        if freq < min_freq:
            continue
        vocab[word] = len(vocab)
    return vocab
