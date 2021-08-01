import argparse
import numpy as np
import torch
from utils import seed_everything
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import lr_scheduler, Adam
from dataset import LoadImageDataset
import train.training as normal_training
from models.unet import UNet
from models.linknet import LinkNet
from models.deeplab import DeepLabV3
from train.curriculum_kd import training as curriculum_kd_training
from train.curriculum_kd import CurricullumKD_Loss
import json

def main():
    parser = argparse.ArgumentParser(description='Curriculum KD')
    parser.add_argument('--data', default='bus', help="options:[neelde, bus]")
    parser.add_argument('--img_size', default=224, type=int,help='img size')
    parser.add_argument('--batch_size', default=16, type=int,help='mini-batch size')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--model_name', default='CKD', help="options:[LinkNet, UNet, CKD, KD, DeepLab]")
    parser.add_argument('--model_path', default='trained_model2/ce_cda_version3.100.pth', help="path where model is to be saved")
    parser.add_argument('--teacher_path', default='trained_model/unet.pth', help="path to pretrained teacher model")
    parser.add_argument('--weights_to', default="ce", help="options:[ce, kld, both]")
    parser.add_argument('--temp', default=4, type=float,help='temperature')
    parser.add_argument('--alpha', default=0.3, type=float,help='alpha')
    parser.add_argument('--lr', default=0.003, type=float,help='initial learning rate')

    args = parser.parse_args()

    with open(args.data+"/data.npy", 'rb') as f:
        train_paths = np.load(f)
        val_paths = np.load(f)
        f.close()

    APPLY_TRANSFORM = True if args.data=="needle" else False
    train_data = LoadImageDataset(train_paths, args.img_size, apply_transform=APPLY_TRANSFORM)
    val_data = LoadImageDataset(val_paths, args.img_size, apply_transform=False)

    dataloaders = {
        'train': DataLoader(train_data, batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
    }
    
    if args.model_name in ['UNet', 'DeepLab', 'LinkNet']:

        if args.model_name == 'UNet':
            student_model = UNet(2)

        elif args.model_name == 'DeepLab':
            student_model = DeepLabV3(2)

        elif args.model_name == 'LinkNet':
            student_model = LinkNet(2)

        student_model = student_model.cuda()
        loss_func = nn.CrossEntropyLoss()
        training = normal_training

    elif args.model_name in ['CKD', 'KD']:
        teacher_model = UNet(2)
        student_model = UNet(2)

        teacher_model = teacher_model.cuda()
        student_model = student_model.cuda()
        teacher_model.load_state_dict(torch.load(args.teacher_path))

        WEIGHTS_TO = 'kd' if args.model_name=="KD" else args.weights_to
        loss_func = CurricullumKD_Loss(WEIGHTS_TO, args.alpha, args.temp)
        training = curriculum_kd_training

    optimizer_ft = Adam(student_model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-08)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    if args.model_name in ['UNet', 'DeepLab', 'LinkNet']:
        student_model, losses = training(student_model, dataloaders, loss_func, optimizer_ft, exp_lr_scheduler, num_epochs=args.epochs)
    else:
        student_model, losses = training(teacher_model, student_model, dataloaders, loss_func, optimizer_ft, exp_lr_scheduler, num_epochs=args.epochs)
    torch.save(student_model.state_dict(), args.model_path)

    loss_file = open(args.model_path[:-4]+".json", "w")
    json.dump(losses, loss_file)
    loss_file.close()

if __name__ == '__main__':
    torch.cuda.set_device(1)
    seed_everything()
    main()