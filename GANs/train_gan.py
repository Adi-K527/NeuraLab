#-------------IMPORTS------------------#
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader


#-------------HYPERPARAMS------------------#
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 100
TEST_SIZE = 0.35
INPUT_DIM = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


#-------------STANDARD GAN MODEL------------------#
class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.block1 = nn.Sequential(
        nn.Linear(784, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2)
    )
    self.block2 = nn.Sequential(
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.LeakyReLU(0.2)
    )
    self.block3 = nn.Sequential(
        nn.Linear(128, 1), nn.Dropout(0.2), nn.Sigmoid()
    )

  def forward(self, x):
    x1 = self.block1(x)
    x2 = self.block2(x1)
    x3 = self.block3(x2)
    return x3


class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.block1 = nn.Sequential(
        nn.Linear(10, 128), nn.LeakyReLU(0.2)
    )
    self.block2 = nn.Sequential(
        nn.Linear(128, 256), nn.LeakyReLU(0.2)
    )
    self.block3 = nn.Sequential(
        nn.Linear(256, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2)
    )
    self.block4 = nn.Sequential(
        nn.Linear(512, 784), nn.Tanh()
    )

  def forward(self, x):
    x1 = self.block1(x)
    x2 = self.block2(x1)
    x3 = self.block3(x2)
    x4 = self.block4(x3)
    return x4


#-------------DCGAN MODEL------------------#
class DCDiscriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.block1 = nn.Sequential(
        nn.Conv2d(1, 128, 4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2)
    )
    self.block2 = nn.Sequential(
        nn.Conv2d(128, 64, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2)
    )
    self.block3 = nn.Sequential(
        nn.Conv2d(64, 32, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2)
    )
    self.block4 = nn.Sequential(
        nn.Conv2d(32, 16, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.LeakyReLU(0.2)
    )
    self.block5 = nn.Sequential(
        nn.Conv2d(16, 1, 2, stride=2, padding=0, bias=False),
        nn.Dropout(0.2),
        nn.Sigmoid()
    )

  def forward(self, x):
    x = x.view((x.shape[0], -1, 28, 28))
    x1 = self.block1(x)
    x2 = self.block2(x1)
    x3 = self.block3(x2)
    x4 = self.block4(x3)
    x5 = self.block5(x4)

    return x5.view((x5.shape[0], 1))


class DCGenerator(nn.Module):
  def __init__(self):
    super().__init__()

    self.block1 = nn.Sequential(
        nn.ConvTranspose2d(INPUT_DIM, 2048, 3, stride=1, padding=0, bias=False),
        nn.ReLU()
    )
    self.block2 = nn.Sequential(
        nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU()
    )
    self.block3 = nn.Sequential(
        nn.ConvTranspose2d(1024, 512, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.block4 = nn.Sequential(
        nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )
    self.block5 = nn.Sequential(
        nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )
    self.block6 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    self.block7 = nn.Sequential(
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU()
    )
    self.block8 = nn.Sequential(
        nn.Conv2d(32, 16, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(16),
        nn.ReLU()
    )
    self.block9 = nn.Sequential(
        nn.Conv2d(16, 8, 4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(8),
        nn.ReLU()
    )
    self.block10 = nn.Sequential(
        nn.Conv2d(8, 6, 4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(6),
        nn.ReLU()
    )
    self.block11 = nn.Sequential(
        nn.Conv2d(6, 4, 4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU()
    )
    self.block12 = nn.Sequential(
        nn.Conv2d(4, 3, 4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(3),
        nn.ReLU()
    )
    self.block13 = nn.Sequential(
        nn.Conv2d(3, 2, 4, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(2),
        nn.ReLU()
    )
    self.block14 = nn.Sequential(
        nn.Conv2d(2, 1, 4, stride=1, padding=0, bias=False),
        nn.Tanh()
    )

  def forward(self, x):
    x  = x.view((x.shape[0], x.shape[1], 1, 1))
    x1 = self.block1(x)
    x2 = self.block2(x1)
    x3 = self.block3(x2)
    x4 = self.block4(x3)
    x5 = self.block5(x4)
    x6 = self.block6(x5)
    x7 = self.block7(x6)
    x8 = self.block8(x7)
    x9 = self.block9(x8)
    x10 = self.block10(x9)
    x11 = self.block11(x10)
    x12 = self.block12(x11)
    x13 = self.block13(x12)
    x14 = self.block14(x13)

    return x14


#-------------DATA PREPROCESSING------------------#
data = pd.read_csv("sample_data/mnist_train_small.csv", header=None)
X = data.drop(columns=[0])
y = data[0]

class MNISTData(Dataset):
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __len__(self):
    return self.x.shape[0]

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]
  
X = torch.tensor(X.to_numpy()).float().to(DEVICE)
y = torch.tensor(y).float().to(DEVICE)

X_train = X[:int(X.shape[0] * TEST_SIZE)]
X_test  = X[int(X.shape[0]  * TEST_SIZE):]
y_train = y[:int(y.shape[0] * TEST_SIZE)]
y_test  = y[int(y.shape[0]  * TEST_SIZE):]

train_data = DataLoader(MNISTData(X_train, y_train),
                        batch_size=BATCH_SIZE, shuffle=True)

test_data  = DataLoader(MNISTData(X_test, y_test),
                        batch_size=BATCH_SIZE, shuffle=True)


#-------------MODEL TRAINING------------------#
disc = Discriminator().to(DEVICE)
gen  = Generator().to(DEVICE)

# Seperate optimizers for generator and discriminator
d_optim = torch.optim.AdamW(disc.parameters(), lr=LEARNING_RATE)
g_optim = torch.optim.AdamW(gen.parameters(),  lr=LEARNING_RATE)

criterion = nn.BCELoss()


def train_disc(x_real, x_fake):
  real_loss = criterion(disc(x_real), torch.ones(x_real.shape[0],  1).to(DEVICE))
  fake_loss = criterion(disc(x_fake), torch.zeros(x_real.shape[0], 1).to(DEVICE))
  loss = real_loss + fake_loss

  d_optim.zero_grad()
  loss.backward()
  d_optim.step()

  return round(loss.item(), 2)


def train_gen(x):
  fakes = disc(gen(x))
  loss  = criterion(fakes, torch.ones(fakes.shape[0], 1).to(DEVICE))

  g_optim.zero_grad()
  loss.backward()
  g_optim.step()

  return round(loss.item(), 2)


def train_epoch():
  gen_lossi  = []
  disc_lossi = []

  for idx, (x,y) in enumerate(train_data):
    if x.shape[0] != BATCH_SIZE:
      continue

    # generates batch of random noise to generate image batches from
    gen_input = torch.randn((BATCH_SIZE, INPUT_DIM)).clamp(-1, 1).to(DEVICE)
    with torch.no_grad():
      x_fake = gen(gen_input)

    disc_loss = train_disc(x, x_fake)
    gen_loss  = train_gen(gen_input)

    gen_lossi.append(gen_loss)
    disc_lossi.append(disc_loss)

    if idx % 10 == 0:
      print(f"ITERATION: {idx}  |  GEN LOSS: {gen_loss}  |  DISC LOSS: {disc_loss}")

  return gen_lossi, disc_lossi

gen_lossi  = []
disc_lossi = []

for epoch in range(EPOCHS):
  print(f"\n\n------------- EPOCH {epoch} -------------")
  g_losses, d_losses = train_epoch()
  gen_lossi.extend(g_losses)
  disc_lossi.extend(d_losses)


#-------------GENERATE Images------------------#
plt.figure(figsize=(10, 10))

for i in range(25):
  plt.subplot(5, 5, i + 1)
  gen.eval()
  plt.imshow(gen(torch.randn((1, INPUT_DIM)).clamp(-1, 1).to(DEVICE)).view((28, 28)).cpu().detach().numpy())

plt.tight_layout()