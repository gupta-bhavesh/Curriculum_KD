from models.myVersion import MyVersion_Loss
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
from models.myVersion import training as my_version_training
import json

#   Seeding things for reproducibility
seed_everything()
DATA_PATH = "data.npy"
IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 100
APPLY_TRANSFORM = False     #   Flip images horizontally or not...
torch.cuda.set_device(1)    #   GPU No. to be used

MODEL_NAME = "Curriculum KD" #   UNet / DeepLab / LinkNet / Curriculum KD / Version 1 / My Version
OPERATION = "Train" #   Train or Test

STUDENT_MODEL_PATH = "trained_model2/ce_cda_version3.100.pth"  #   Main Model which is to be trained or tested...
TEACHER_MODEL_PATH = "trained_model/unet.pth"  #   Ignore if not applying Curricullum KD...

#   Loading paths of training and validation images 
with open(DATA_PATH, 'rb') as f:
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

    elif MODEL_NAME in ['Curriculum KD', 'Version 1', 'Version 2', 'Version 3', 'Version 4', 'My Version']:
        teacher_model = UNet(2)
        student_model = UNet(2)

        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()
        teacher_model.load_state_dict(torch.load(TEACHER_MODEL_PATH))

        if MODEL_NAME == 'Curriculum KD':
            TEMP = 4.0
            WEIGHTS_TO = "ce"
            ALPHA=0.3
            loss_func = CurricullumKD_Loss(WEIGHTS_TO, ALPHA, TEMP)
            training = curriculum_kd_training

        elif MODEL_NAME == 'Version 1':
            TEMP = 4
            ALPHA = 0.3
            WEIGHTS_TO = "ce"
            MU = 1.4
            loss_func = Version1_Loss(weights_to=WEIGHTS_TO, temp=TEMP, alpha=ALPHA)
            training = version1_training

        elif MODEL_NAME in ['Version 2', 'Version 3', 'Version 4']:
            TEMP = 3
            loss_func = Version234_Loss(TEMP, MODEL_NAME)
            training = version234_training
        
        elif MODEL_NAME == 'My Version':
            TEMP = 4
            WEIGHTS_TO = "ce"
            ALPHA=0.3

            loss_func = MyVersion_Loss(weights_to = WEIGHTS_TO, alpha=ALPHA, temp=TEMP)
            training = my_version_training

    optimizer_ft = Adam(student_model.parameters(), lr=0.003, betas=(0.9, 0.99), eps=1e-08)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    student_model, losses = training(teacher_model, student_model, dataloaders, loss_func, optimizer_ft, exp_lr_scheduler, num_epochs=NUM_EPOCHS)
    torch.save(student_model.state_dict(), STUDENT_MODEL_PATH)

    loss_file = open(STUDENT_MODEL_PATH[:-4]+".json", "w")
    json.dump(losses, loss_file)
    loss_file.close()

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