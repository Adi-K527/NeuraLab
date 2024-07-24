#-------------IMPORTS------------------#
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from zipfile import ZipFile
from PIL import Image


#-------------HYPERPARAMS------------------#
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"
EPOCHS = 5
LEARNING_RATE = 3e-4


#-------------FETCH DATA------------------#
kaggle_json = {
    "username"  : "",
    "key"       : ""
}

with open('kaggle.json', 'w') as fp:
    json.dump(kaggle_json, fp)

path = '/content/brain-tumor-image-dataset-semantic-segmentation.zip'
with ZipFile(path, 'r') as zip:
  zip.extractall()

with open('/content/train/_annotations.coco.json','r') as file:
    train_data_json = json.load(file)

with open('/content/valid/_annotations.coco.json','r') as file:
    valid_data_json = json.load(file)

with open('/content/test/_annotations.coco.json','r') as file:
    test_data_json = json.load(file)


#-------------UNET MODEL------------------#

def double_conv(in_channels, out_channels):
  return nn.Sequential(
      nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
      nn.ReLU(inplace=True),
  )


def crop_tensor(orig, target_shape):
  start = (orig.shape[2] - target_shape) // 2
  end = orig.shape[2] - start
  if end - start > target_shape:
    end -= 1
  return orig[:, :, start:end, start:end]


class UNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.down_conv_1 = double_conv(1, 64)
    self.down_conv_2 = double_conv(64, 128)
    self.down_conv_3 = double_conv(128, 256)
    self.down_conv_4 = double_conv(256, 512)
    self.down_conv_5 = double_conv(512, 1024)

    self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
    self.up_conv_1 = double_conv(1024, 512)
    self.up_trans_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
    self.up_conv_2 = double_conv(512, 256)
    self.up_trans_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
    self.up_conv_3 = double_conv(256, 128)
    self.up_trans_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
    self.up_conv_4 = double_conv(128, 64)

    self.out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)


  def forward(self, x):
    x1 = self.down_conv_1(x)
    x2 = self.max_pool_2x2(x1)
    x3 = self.down_conv_2(x2)
    x4 = self.max_pool_2x2(x3)
    x5 = self.down_conv_3(x4)
    x6 = self.max_pool_2x2(x5)
    x7 = self.down_conv_4(x6)
    x8 = self.max_pool_2x2(x7)
    x9 = self.down_conv_5(x8)

    x = self.up_trans_1(x9)
    x = torch.cat((x, crop_tensor(x7, x.shape[2])), 1)
    x = self.up_conv_1(x)
    x = self.up_trans_2(x)
    x = torch.cat((x, crop_tensor(x5, x.shape[2])), 1)
    x = self.up_conv_2(x)
    x = self.up_trans_3(x)
    x = torch.cat((x, crop_tensor(x3, x.shape[2])), 1)
    x = self.up_conv_3(x)
    x = self.up_trans_4(x)
    x = torch.cat((x, crop_tensor(x1, x.shape[2])), 1)
    x = self.up_conv_4(x)

    return self.out(x)


#-------------DATA PREPROCESSING------------------#
class ImageDataset(Dataset):
  def __init__(self, x, y):
    train_files = x
    train_masks = y


    self.x = torch.zeros((len(train_files), 224, 224))
    self.y = torch.zeros((len(train_files), 224, 224))
    self.file_names = x

    for i in range(len(train_files)):
      img = Image.open(train_files[i])
      img = img.convert("L").resize((224, 224))
      self.x[i] = torch.tensor(np.array(img))

      xi_mask = Image.fromarray(np.uint8(train_masks[i]))
      xi_mask = xi_mask.resize((224, 224))
      self.y[i] = torch.tensor(np.array(xi_mask))

    self.x = self.x.unsqueeze(1)
    self.y = self.y.unsqueeze(1)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx], self.file_names[idx]

train_data = pd.merge(pd.DataFrame(train_data_json["images"]),
                      pd.DataFrame(train_data_json["annotations"]), on="id")
train_data["file_name"] = train_data["file_name"].apply(lambda x: "train/" + x)

valid_data = pd.merge(pd.DataFrame(valid_data_json["images"]),
                      pd.DataFrame(valid_data_json["annotations"]), on="id")
valid_data["file_name"] = valid_data["file_name"].apply(lambda x: "valid/" + x)

test_data = pd.merge(pd.DataFrame(test_data_json["images"]),
                    pd.DataFrame(test_data_json["annotations"]), on="id")
test_data["file_name"] = test_data["file_name"].apply(lambda x: "test/" + x)


def create_masks(data, idx):
  temp = data.iloc[idx]
  points = torch.tensor(temp["segmentation"][0]).view((-1, 2))[:4]
  mat = torch.zeros((640, 640))

  mat[int(min(points[:,1]).item()):int(max(points[:,1]).item()),
      int(min(points[:,0]).item()):int(max(points[:,0]).item())] = 1

  return mat

arr = []
for data in [train_data, valid_data, test_data]:
  arr.append(torch.zeros((data.shape[0], 640, 640)))
  for i in range(data.shape[0]):
    mask = create_masks(data, i)
    arr[-1][i] = mask

train_masks, valid_masks, test_masks = arr


train_img_paths = ["/content/" + train_data.iloc[i]["file_name"] for i in range(train_data.shape[0])]
valid_img_paths = ["/content/" + valid_data.iloc[i]["file_name"] for i in range(valid_data.shape[0])]
test_img_paths = ["/content/"  +  test_data.iloc[i]["file_name"] for i in range(test_data.shape[0])]

train_data = DataLoader(ImageDataset(train_img_paths, train_masks), BATCH_SIZE, shuffle=True)
valid_data = DataLoader(ImageDataset(valid_img_paths, valid_masks), BATCH_SIZE, shuffle=True)
test_data = DataLoader(ImageDataset(test_img_paths,   test_masks),  BATCH_SIZE, shuffle=True)


#-------------MODEL TRAINING------------------#
model = UNet()
optim = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

def train_epoch(model, train_data):
  model.train()
  lossi = []
  for x,y,_ in train_data:
    y = y.to(DEVICE)
    x = x.to(DEVICE)
    output = model(x)

    loss = F.binary_cross_entropy_with_logits(output, y)
    optim.zero_grad()
    loss.backward()
    optim.step()

    lossi.append(loss.item())

  return torch.tensor(lossi).mean()


for epoch in range(EPOCHS):
  loss = train_epoch(model, train_data)
  print(f"EPOCH: {epoch}  |  LOSS: {loss}")