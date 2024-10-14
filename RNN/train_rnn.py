#-------------IMPORTS------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd


#-------------HYPERPARAMS------------------#
CONTEXT_LENGTH = 8
EPOCHS = 20
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
HIDDEN_SIZE = 32


#-------------RNN MODEL------------------#
class RNNCell(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super().__init__()
        self.Wxh = nn.Linear(in_dim, hidden_dim)
        self.Whh = nn.Linear(hidden_dim, hidden_dim)
        self.Who = nn.Linear(hidden_dim, out_dim)

    def forward(self, xt, ht):
        xh = self.Wxh(xt)
        hidden_state = self.Whh(ht)
        hidden_state = F.tanh(hidden_state + xh)
        output = self.Who(hidden_state)
        return (hidden_state, output)


class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = RNNCell(in_dim, hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x):
        h_t_minus_1 = torch.zeros((1, HIDDEN_SIZE))
        h_t = torch.zeros((1, HIDDEN_SIZE))
        res = []
        for t in range(x.shape[1]):
            h_t, output = self.cell(x[:,t].view(-1, 1), h_t_minus_1)
            res.append(h_t)
            h_t_minus_1 = h_t

        res = torch.stack(res)
        
        final_out = res[-1]
        output = self.out_lin(final_out)
        return output


#-------------DATA PREPROCESSING------------------#
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

data = torch.tensor(pd.read_csv("data.txt", header=None).values)

arr = [data[i:i+CONTEXT_LENGTH] for i in range(len(data) - CONTEXT_LENGTH)]
x = torch.hstack(arr).transpose(0, 1)
y = torch.tensor(data[CONTEXT_LENGTH:len(data)])

size = int(0.8 * x.shape[0])
X_train = x[:size]
y_train = y[:size]

X_test = x[size:]
y_test = y[size:]
    
train_dataset = DataLoader(TimeSeriesDataset(X_train, y_train), 
                                      BATCH_SIZE, shuffle=True)
test_dataset = DataLoader(TimeSeriesDataset(X_test, y_test), 
                                   BATCH_SIZE, shuffle=True)


#-------------MODEL TRAINING------------------#
model = RNN(CONTEXT_LENGTH, HIDDEN_SIZE)
optim = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

def train_epoch():
    lossi = []
    for batch in train_dataset:
        x, y = batch
        output = model(x.float())
        loss = F.mse_loss(output, y.float())
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        lossi.append(loss.item())
    
    return torch.tensor(lossi).mean().item()

@torch.no_grad()
def val_epoch():
    lossi = []
    for batch in test_dataset:
        x, y = batch
        output = model(x.float())
        loss = F.mse_loss(output, y.float())
        lossi.append(loss.item())
    
    return torch.tensor(lossi).mean().item()

train_lossi = []
test_lossi = []
for epoch in range(EPOCHS):
  train_loss = train_epoch()
  test_loss = val_epoch()

  train_lossi.append(train_loss)
  test_lossi.append(test_loss)

  print(f"EPOCH: {epoch} |  TRAIN-LOSS: {train_loss} |  VAL-LOSS: {test_loss}")