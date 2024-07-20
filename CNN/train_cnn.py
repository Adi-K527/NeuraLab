#-------------IMPORTS------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score


#-------------HYPERPARAMS------------------#
EPOCHS = 5
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
KERNEL_DIM = 3
POOL_DIM = 2


#-------------CNN MODEL------------------#
class Conv2dLayer(nn.Module):
    def __init__(self, in_dim, kernel_size):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        self.kernel = nn.Linear(kernel_size[0], kernel_size[0])

    def forward(self, x):
        x = x.unsqueeze(1)
        output_size = x.shape[-1] - self.kernel_size[0] + 1
        windows = F.unfold(x, x.shape[-1] - self.kernel_size[0] + 1)
        windows = windows.view((windows.shape[0], windows.shape[1], self.kernel_size[0], self.kernel_size[0]))

        convs = self.kernel(windows).sum(dim=-1).sum(dim=-1)
        output = convs.view((convs.shape[0], output_size, output_size))

        return output


class AvgPool2dLayer(nn.Module):
    def __init__(self, pool_size):
        super().__init__()
        self.kernel_size = pool_size

    def forward(self, x):
        x = x.unsqueeze(1)
        window_size = x.shape[-1] - self.kernel_size[0] + 1

        windows = F.unfold(x, kernel_size=window_size).mean(dim=-1)
        output = windows.view((x.shape[0], window_size, window_size))[:, ::self.kernel_size[0], ::self.kernel_size[1]]
        
        return output


class CNN(nn.Module):
    def __init__(self, in_dim, conv1_kernel_size, conv_pool_size):
        super().__init__()
        self.conv1 = Conv2dLayer(in_dim, (conv1_kernel_size, conv1_kernel_size))
        self.pool  = AvgPool2dLayer((conv_pool_size, conv_pool_size))
        self.conv2 = Conv2dLayer(in_dim, (conv1_kernel_size, conv1_kernel_size))
        self.lin   = nn.Linear(25, 10)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.pool(out1)
        out3 = self.conv2(out2)
        out4 = self.pool(out3)
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


X_train = pd.read_csv("data/train_data.csv").drop(['Unnamed: 0', 'Y'], axis=1)
y_train = pd.read_csv("data/train_data.csv")['Y']
X_test = pd.read_csv("data/test_data.csv").drop(['Unnamed: 0', 'Y'], axis=1)
y_test = pd.read_csv("data/test_data.csv")['Y']

X_train = torch.tensor(X_train.to_numpy()).view((-1, 28, 28)) / 255.0
y_train = torch.tensor(y_train.values)
X_test = torch.tensor(X_test.to_numpy()).view((-1, 28, 28)) / 255.0
y_test = torch.tensor(y_test.values)

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
