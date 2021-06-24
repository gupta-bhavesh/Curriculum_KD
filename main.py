import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import lr_scheduler, Adam
from utils import seed_everything
from dataset import LoadImageDataset
from testing import testing
from training import training as normal_training
from models.unet import UNet
from models.linknet import LinkNet
from models.deeplab import DeepLabV3
from models.version234 import training as version234_training
from models.version234 import Version234_Loss
from models.version1 import training as version1_training
from models.version1 import Version1_Loss
from models.curriculum_kd import training as curriculum_kd_training
from models.curriculum_kd import CurricullumKD_Loss

#   Seeding things for reproducibility
seed_everything()

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 100
torch.cuda.set_device(1)    #   GPU No. to be used
APPLY_TRANSFORM = False     #   Flip images horizontally or not...

MODEL_NAME = "Version 1" #   UNet / DeepLab / LinkNet / KD / Curriculum / Curriculum KD / Version 1 / Version 2 / Version 3 / Version 4

OPERATION = "Train" #   Train or Test

STUDENT_MODEL_PATH = "trained_model/segmentation_unet_version1_kld_pixelwise.pth"  #   Main Model which is to be trained or tested...
TEACHER_MODEL_PATH = "trained_model/segmentation_unet_weights_224.pth"  #   Ignore if not applying Curricullum KD...

#   Loading paths of training and validation images 
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

if OPERATION == "Train":

    if MODEL_NAME in ['UNet', 'DeepLab', 'LinkNet']:

        if MODEL_NAME == 'UNet':
            student_model = UNet(2)

        elif MODEL_NAME == 'DeepLab':
            student_model = DeepLabV3(2)

        elif MODEL_NAME == 'LinkNet':
            student_model = LinkNet(2)

        student_model = student_model.cuda()
        loss_func = nn.CrossEntropyLoss()
        training = normal_training

    elif MODEL_NAME in ['KD', 'Curriculum KD', 'Version 1', 'Version 2', 'Version 3', 'Version 4']:
        teacher_model = UNet(2)
        student_model = UNet(2)
        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()
        teacher_model.load_state_dict(torch.load(TEACHER_MODEL_PATH))

        if MODEL_NAME == 'KD':
            TEMP = 3
            WEIGHTS_TO = "both"
            ALPHA=0.5

        elif MODEL_NAME == 'Curriculum':
            TEMP = 3
            ALPHA=0.5

        elif MODEL_NAME == 'Curriculum KD':
            TEMP = 3
            WEIGHTS_TO = "both"
            ALPHA=0.5
            loss_func = CurricullumKD_Loss(WEIGHTS_TO, ALPHA, TEMP)
            training = curriculum_kd_training

        elif MODEL_NAME == 'Version 1':
            TEMP = 3
            WEIGHTS_TO = "kld"
            MU = 1.05
            loss_func = Version1_Loss(WEIGHTS_TO, TEMP)
            training = version1_training

        elif MODEL_NAME in ['Version 2', 'Version 3', 'Version 4']:
            TEMP = 3
            loss_func = Version234_Loss(TEMP, MODEL_NAME)
            training = version234_training

    
    optimizer_ft = Adam(student_model.parameters(), lr=0.003, betas=(0.9, 0.99), eps=1e-08)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    student_model = training(teacher_model, student_model, dataloaders, loss_func, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS, mu=MU)
    torch.save(student_model.state_dict(), STUDENT_MODEL_PATH)

elif OPERATION == "Test":
    if MODEL_NAME in ['UNet', 'DeepLab', 'LinkNet']:

        if MODEL_NAME == 'UNet':
            student_model = UNet(2)

        elif MODEL_NAME == 'DeepLab':
            student_model = DeepLabV3(2)

        elif MODEL_NAME == 'LinkNet':
            student_model = LinkNet(2)

        student_model = student_model.cuda()

    student_model.load_state_dict(torch.load(STUDENT_MODEL_PATH))
    testing(student_model, dataloaders)

else:
    raise Exception("Must be one of Train or Test")