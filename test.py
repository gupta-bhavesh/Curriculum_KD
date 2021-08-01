import torch
import numpy as np
from dataset import LoadImageDataset
from torch.utils.data import DataLoader
from models.unet import UNet
from models.linknet import LinkNet
from models.deeplab import DeepLabV3
from needle.testing import testing as needle_testing
from bus.testing import testing as bus_testing
import argparse

def main():
    parser = argparse.ArgumentParser(description='Curriculum KD')
    parser.add_argument('--data', default='bus', help="options:[neelde, bus]")
    parser.add_argument('--img_size', default=224, type=int,help='img size')
    parser.add_argument('--batch_size', default=16, type=int,help='mini-batch size')
    parser.add_argument('--model_name', default='UNet', help="options:[LinkNet, UNet, CKD, KD, DeepLab]")
    parser.add_argument('--model_path', default='trained_model/unet.pth', help="Path to trained model weights")

    args = parser.parse_args()

    with open(args.data+"_data.npy", 'rb') as f:
        val_paths = np.load(f)
        f.close()

    val_data = LoadImageDataset(val_paths, args.img_size, apply_transform=False)
    dataloaders = {'val': DataLoader(val_data, batch_size=args.batch_size, shuffle=True)}

    if args.model_name == 'DeepLab':
        seg_model = DeepLabV3(2)
    elif args.model_name == 'LinkNet':
        seg_model = LinkNet(2)
    else:
        seg_model = UNet(2)

    seg_model = seg_model.cuda()
    seg_model.load_state_dict(torch.load(args.model_path))

    if(args.data=="needle"):
        dice, iou, precision, recall, F1 = needle_testing(seg_model, dataloaders)
    elif(args.data=="bus"):
        dice, iou, precision, recall, F1 = bus_testing(seg_model, dataloaders)

if __name__ == '__main__':
    torch.cuda.set_device(1)
    main()