#-------------IMPORTS------------------#
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import seaborn as sns
import pandas as pd
import tensorflow as tf
import numpy as np
from zipfile import ZipFile
from PIL import Image
from sklearn.metrics import accuracy_score


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

model = smp.Unet( # pretrained model for now
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=1,
    classes=1,
    activation=None,
)

model.to(DEVICE)


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
optim = torch.optim.AdamW(model.parameters(), LEARNING_RATE)

def convert_to_tensor(batch):
  batch_tensor = torch.zeros((len(batch), 640, 640, 3))
  for i in range(len(batch)):
    batch_tensor[i] = torch.tensor(np.array(Image.open(batch[i])))

  return batch_tensor

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