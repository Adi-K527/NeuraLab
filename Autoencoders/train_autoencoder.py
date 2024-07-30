#-------------IMPORTS------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt


#-------------HYPERPARAMS------------------#
LEARNING_RATE = 3e-3
EPOCHS = 5
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_FACTOR = 10


#-------------AutoEncoder MODEL------------------#
class AutoEncoder(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # encoder
        self.e1 = nn.Linear(in_dim, 256)
        self.norme1 = nn.BatchNorm1d(256)
        self.e2 = nn.Linear(256, 64)
        self.norme2 = nn.BatchNorm1d(64)
        self.e3 = nn.Linear(64, 8)
        self.norme3 = nn.BatchNorm1d(8)
        self.relu = nn.ReLU()

        # decoder
        self.d1 = nn.Linear(8, 64)
        self.normd1 = nn.BatchNorm1d(64)
        self.d2 = nn.Linear(64, 256)
        self.normd2 = nn.BatchNorm1d(256)
        self.d3 = nn.Linear(256, in_dim)
        self.normd3 = nn.BatchNorm1d(in_dim)

    def forward(self, x):
        x = x.view((BATCH_SIZE, -1))
        x = self.norme1(self.e1(x))
        x = self.relu(x)
        x = self.norme2(self.e2(x))
        x = self.relu(x)
        x = self.norme3(self.e3(x))
        x = self.relu(x)

        x = self.normd1(self.d1(x))
        x = self.relu(x)
        x = self.normd2(self.d2(x))
        x = self.relu(x)
        x = self.normd3(self.d3(x))
        x = self.relu(x)

        return x


#-------------Convolutional AutoEncoder MODEL------------------#
class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, in_dim=None):
      super().__init__()

      self.enc_block_1 = nn.Sequential(
          nn.Conv2d(1, 16, 3, padding=1),
          nn.BatchNorm2d(16),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
      )
      self.enc_block_2 = nn.Sequential(
          nn.Conv2d(16, 4, 3, padding=1),
          nn.BatchNorm2d(4),
          nn.ReLU(),
          nn.MaxPool2d(2, 2),
      )
      self.dec_block_1 = nn.Sequential(
          nn.ConvTranspose2d(4, 16, 2, stride=2),
          nn.BatchNorm2d(16),
          nn.ReLU()
      )
      self.dec_block_2 = nn.Sequential(
          nn.ConvTranspose2d(16, 1, 2, stride=2),
          nn.ReLU()
      )

    def forward(self, x):
      x = self.enc_block_1(x)
      x = self.enc_block_2(x)

      x = self.dec_block_1(x)
      x = self.dec_block_2(x)

      return x


#-------------DATA PREPROCESSING------------------#
class MnistDataset(Dataset):
    def __init__(self, data):
        self.x = data + (NOISE_FACTOR * torch.randn_like(data))
        self.y = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

train_data_df = pd.read_csv("/data/train_data.csv")
test_data_df = pd.read_csv("/data/test_data.csv")

train_data_df.drop(['Unnamed: 0', 'Y'], axis=1, inplace=True)
test_data_df.drop(['Unnamed: 0', 'Y'], axis=1, inplace=True)

train_data_tensor = torch.tensor(train_data_df.to_numpy(), dtype=float)
test_data_tensor = torch.tensor(test_data_df.to_numpy(), dtype=float)
    
train_data = DataLoader(MnistDataset(train_data_tensor), batch_size=BATCH_SIZE, shuffle=True)
test_data  = DataLoader(MnistDataset(test_data_tensor),  batch_size=BATCH_SIZE, shuffle=True)


#-------------AutoEncoder Epoch Training Functions------------------#
def train_epoch(model, optim, train_data):
    lossi = []
    for x,y in train_data:
      if x.shape[0] != BATCH_SIZE:
        continue
      x = x.view((BATCH_SIZE, 1, 28, 28))
      x, y = x.float().to(DEVICE), y.to(DEVICE)

      output = model(x).view(BATCH_SIZE, -1)
      loss = ((output - y) ** 2).mean()

      optim.zero_grad()
      loss.backward()
      optim.step()

      lossi.append(loss.item())

    return torch.tensor(lossi).mean()

@torch.no_grad()
def test_epoch(model, test_data):
    lossi = []
    for x,y in test_data:
      if x.shape[0] != BATCH_SIZE:
        continue
      x = x.view((BATCH_SIZE, 1, 28, 28))
      x, y = x.float().to(DEVICE), y.to(DEVICE)
      output = model(x).view(BATCH_SIZE, 1, -1)
      loss = ((output - y) ** 2).mean()
      lossi.append(loss.item())

    return torch.tensor(lossi).mean()

def train_model(model, optim, train_data, test_data):
    losses = []
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, optim, train_data)
        test_loss = test_epoch(model, test_data)
        losses.append([train_loss.item(), test_loss.item()])

        print(f"EPOCH: {epoch} | TRAIN_LOSS: {train_loss.item()} | TEST_LOSS: {test_loss.item()}")
    return losses


#-------------AutoEncoder TRAINING------------------#
model = AutoEncoder(784).to(DEVICE)
optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
train_model(model, optim, train_data, test_data)


#-------------Convolutional AutoEncoder TRAINING------------------#
model = ConvolutionalAutoEncoder(784).to(DEVICE)
optim = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
train_model(model, optim, train_data, test_data)
