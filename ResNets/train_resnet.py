#-------------IMPORTS------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd


#-------------HYPERPARAMS------------------#
LEARNING_RATE = 3e-4
BATCH_SIZE = 16
BLOCK_SIZE = 3
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


#-------------RNN MODEL------------------#
class Block(nn.Module):
  def __init__(self, d0, d1, d2, d3):
    super().__init__()
    self.full = nn.Sequential(
        nn.Linear(d0, d1), nn.BatchNorm1d(d1), nn.ReLU(),
        nn.Linear(d1, d2), nn.BatchNorm1d(d2), nn.ReLU(),
        nn.Linear(d2, d3), nn.BatchNorm1d(d3), nn.ReLU()
    )

  def forward(self, x):
    return self.full(x)


class ResNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(784, 700)
    self.b1 = Block(700, 650, 600, 550)
    self.b2 = Block(550, 500, 450, 400)
    self.b3 = Block(400, 350, 300, 250)
    self.b4 = Block(250, 200, 150, 100)
    self.b5 = Block(100,  64,  32,  16)
    self.l2 = nn.Linear(16, 10)

  def forward(self, x):
    o1 = self.l1(x)
    o2 = self.b1(o1 +  x[:, 42:-42])
    o3 = self.b2(o2 + o1[:, 75:-75])
    o4 = self.b3(o3 + o2[:, 75:-75])
    o5 = self.b4(o4 + o3[:, 75:-75])
    o6 = self.b5(o5 + o4[:, 75:-75])
    output = self.l2(o6)

    return output


#-------------DATA PREPROCESSING------------------#
def get_data(path):
  data = pd.read_csv(path).drop(labels=['Y'], axis=1)
  labels = pd.read_csv(path)['Y']
  data = torch.tensor(data.values)[:, 1::].float()
  data = ((data - data.mean()) / data.std())
  labels = torch.tensor(labels.values).float()
  return data, labels

class MnistData(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return (self.x[idx], self.y[idx])

train_data, train_labels = get_data("train_data.csv")
test_data, test_labels   = get_data("test_data.csv")

train_data = DataLoader(MnistData(train_data, train_labels), BATCH_SIZE, shuffle=True)
test_data  = DataLoader(MnistData(test_data,   test_labels), BATCH_SIZE, shuffle=True)


#-------------MODEL TRAINING------------------#
def train_epoch(data, model, optim):
  lossi = []
  for x, y in data:
    x = x.to(DEVICE)
    y = y.type(torch.LongTensor)
    y = y.to(DEVICE)

    output = model(x.float())
    loss = F.cross_entropy(output, y)

    optim.zero_grad()
    loss.backward()
    optim.step()

    lossi.append(loss.item())

  return torch.tensor(lossi).mean()


def test_epoch(data, model):
  lossi = []
  for x, y in data:
    x = x.to(DEVICE)
    y = y.type(torch.LongTensor)
    y = y.to(DEVICE)

    output = model(x.float())
    loss = F.cross_entropy(output, y)

    lossi.append(loss.item())

  return torch.tensor(lossi).mean()


def train(train_data, test_data, model, optim):
  train_lossi, test_lossi = [], []
  for epoch in range(EPOCHS):
    train_lossi.append(train_epoch(train_data, model, optim))
    test_lossi.append(test_epoch(test_data, model))

    print(f"EPOCH: {epoch}  |  TRAIN_LOSS: {train_lossi[-1]}  |  TEST_LOSS: {test_lossi[-1]}")

  return train_lossi, test_lossi

model = ResNet().to(DEVICE)
optim = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
train_loss_res, test_loss_res = train(train_data, test_data, model, optim)
print(f"-------------------RESNET TRAINING COMPLETE-------------------\n\n")