import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import pandas as pd
from dataset import LoadImageDataset
from torch.utils.data import DataLoader
from models.unet import UNet
from models.linknet import LinkNet
from models.deeplab import DeepLabV3
from testing import testing

IMAGE_SIZE = 224
BATCH_SIZE = 16
APPLY_TRANSFORM = False     #   Flip images horizontally or not...
torch.cuda.set_device(1)    #   GPU No. to be used

with open('data.npy', 'rb') as f:
    train_paths = np.load(f)
    val_paths = np.load(f)

#   Loading Data
train_data = LoadImageDataset(train_paths, IMAGE_SIZE, apply_transform=APPLY_TRANSFORM)
val_data = LoadImageDataset(val_paths, IMAGE_SIZE, apply_transform=False)

dataloaders = {
    'train': DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True),
    'val': DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)
    }

seg_model = UNet(2)
seg_model = seg_model.cuda()
CSV_FILE = 'results2.csv'
results = pd.read_csv(CSV_FILE, index_col=['Model'])

directory = "trained_model/"
# trained_model_files = [i for i in os.listdir(directory) if i.endswith('.pth')]
#trained_model_files.remove('linknet.pth')
#trained_model_files.remove('deeplab.pth')
trained_model_files = ['ce_ckd.pth']
for i in trained_model_files:
    model_path = directory+i
    print(model_path)
    seg_model.load_state_dict(torch.load(model_path))
    print("-"*10 + i[:-4] + "-"*10)

    dice, iou, precision, recall, F1 = testing(seg_model, dataloaders)
    results.loc[i[:-4]] = [dice, iou, precision, recall, F1]
print(results)
results.to_csv(CSV_FILE)