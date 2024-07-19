#-------------IMPORTS------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score


#-------------HYPERPARAMS------------------#
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
HIDDEN_DIM = 64
KERNEL_DIM = 3
POOL_DIM = 2


#-------------RNN MODEL------------------#
class CNN(nn.Module):
    def __init__(self, in_dim, conv1_kernel_size, conv_pool_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, in_dim, (conv1_kernel_size, conv1_kernel_size))
        self.pool1 = nn.AvgPool2d((conv_pool_size, conv_pool_size))
        self.conv2 = nn.Conv2d(in_dim, in_dim, (conv1_kernel_size, conv1_kernel_size))
        self.pool2 = nn.AvgPool2d((conv_pool_size, conv_pool_size))
        self.lin  = nn.Linear(25, len(set(y_train)))

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.pool1(out1)
        out3 = self.conv2(out2)
        out4 = self.pool2(out3)
        out5 = out4.flatten(1)
        output = self.lin(out5)
        return output


#-------------DATA PREPROCESSING------------------#
class MNISTData(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return (self.x[i], self.y[i])

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

train_data = DataLoader(MNISTData(X_train, y_train), BATCH_SIZE, shuffle=True)
test_data = DataLoader(MNISTData(X_test, y_test), BATCH_SIZE, shuffle=True)


#-------------MODEL TRAINING------------------#
model = CNN(BATCH_SIZE, KERNEL_DIM, POOL_DIM)
optim = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

def train_epoch():
    lossi = []
    for x, y in train_data:
        output = model(x)
        loss = F.cross_entropy(output, y)

        optim.zero_grad()
        loss.backward()
        optim.step()

        lossi.append(loss.item())

    return torch.tensor(lossi).mean().item()


@torch.no_grad
def val_epoch():
    lossi = []
    for x, y in test_data:
        output = model(x)
        loss = F.cross_entropy(output, y)
        lossi.append(loss.item())

    return torch.tensor(lossi).mean().item()


train_lossi, val_lossi = [], []
for epoch in range(EPOCHS):
    train_loss = train_epoch()
    val_loss = val_epoch()

    train_lossi.append(train_loss)
    val_lossi.append(val_loss)

    print(f"EPOCH: {epoch} |  TRAIN-LOSS: {train_loss} |  VAL-LOSS: {val_loss}")


#-------------MODEL EVALUATION------------------#
acc = []
for x,y in train_data:
    cur_pred = model(x).argmax(dim=1)
    acc.append(accuracy_score(cur_pred, y))

print(f"Train Accuracy: {torch.tensor(acc).mean() * 100}%")

acc = []
for x,y in test_data:
    cur_pred = model(x).argmax(dim=1)
    acc.append(accuracy_score(cur_pred, y))

print(f"Test Accuracy: {torch.tensor(acc).mean() * 100}%")