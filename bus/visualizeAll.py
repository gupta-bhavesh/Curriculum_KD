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
from utils import max_contour
from torch import nn
import torch.nn.functional as F

def visualizing_results(model, inputs, index, num_images=4):
    was_training = model.training
    model.eval()
    images_so_far = 0
    with torch.no_grad():
        outputs = model(inputs)
        for j in range(16):
            images_so_far += 1
            pred = outputs.cpu().data[j].squeeze()
            pred = F.softmax(pred, dim=0)[1]
            pred = pred.squeeze().cpu().numpy()

            pred[pred>0.5] = 255
            pred[pred<=0.5] = 0
            post_process = max_contour(pred)
            axes[j, index].imshow(post_process, cmap='gray')

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
        model.train(mode=was_training)
        return

IMAGE_SIZE = 224
BATCH_SIZE = 16
APPLY_TRANSFORM = False     #   Flip images horizontally or not...

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

directory = "trained_model/"
trained_model_files = ['unet.pth', 'kd_temp_4.0_alpha_0.3.pth', 'ce_mu_plus_0.1.pth']
num_images = 4
figure, axes = plt.subplots(nrows=num_images, ncols=len(trained_model_files)+2, figsize=(15,3.75*num_images))
inputs, labels = next(iter(dataloaders['val']))
inputs = inputs.cuda()
labels = labels.cuda()
for i in range(num_images):
    img = inputs.cpu().data[i].squeeze()
    label = labels.cpu().data[i].squeeze()
    label = label.squeeze().cpu().numpy()
    axes[i, 0].imshow(np.transpose(img, (1,2,0)), cmap='gray')
    axes[i, 1].imshow(label, cmap='gray')

for index, path in enumerate(trained_model_files):
    print("-"*10 + path[:-4] + "-"*10)
    model_path = directory+path
    seg_model.load_state_dict(torch.load(model_path))
    visualizing_results(seg_model, inputs, index+2, num_images=num_images)

cols = ['Input', 'Prediction']+['UNet', 'KD', 'CKD']
for ax, col in zip(axes[0], cols):
    ax.set_title(col)
plt.show()