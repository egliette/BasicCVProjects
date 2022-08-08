import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class ShelterOutcomeModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embedding_size = 2
        self.embeddings = nn.ModuleList([nn.Embedding(n_categories, self.embedding_size) for n_categories, _ in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.n_classes = 5
        self.lin1 = nn.Linear(self.n_emb + n_cont, 128)
        self.lin2 = nn.Linear(128, 32)
        self.lin3 = nn.Linear(32, self.n_classes)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(32)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def forward(self, x_cat, x_cont):
        x_cat = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x_cat = torch.cat(x_cat, 1)
        x_cat = self.emb_drop(x_cat)
        x_cont = self.bn1(x_cont)
        x = torch.cat([x_cat, x_cont], dim=1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x

def _train_model(model, optim, train_dl):
    model.train()
    total = 0
    sum_loss = 0
    for x_cat, x_cont, y in train_dl:
        current_batch_size = y.shape[0]
        output = model(x_cat, x_cont)
        y  = torch.tensor(y, dtype=torch.long)
        loss = F.cross_entropy(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += current_batch_size
        sum_loss += current_batch_size * loss.item()
    return sum_loss/total

def _val_loss(model, valid_dl):
    model.eval()
    with torch.no_grad():
        total = 0
        sum_loss = 0
        correct = 0
        for x_cat, x_cont, y in valid_dl:
            current_batch_size = y.shape[0]
            output = model(x_cat, x_cont)
            y = torch.tensor(y, dtype=torch.long)
            loss = F.cross_entropy(output, y)
            total += current_batch_size
            sum_loss += current_batch_size * loss.item()
            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == y).item()
    return sum_loss / total, correct / total

def train_loop(model, train_dl, valid_dl, optim, epochs, lr=0.01, wd=0.0):
    logger = logging.getLogger('train')
    for i in range(epochs):
        loss = _train_model(model, optim, train_dl)
        valid_loss, valid_accur = _val_loss(model, valid_dl)
        logger.info(f"Epoch {i}:\tTraining loss: {loss:.3},\tValid loss {valid_loss:.3},\t Accuracy {valid_accur:.3}")

