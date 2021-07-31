import os
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.io.image import ImageReadMode
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize, InterpolationMode
import random

def load_path(path):
    names = os.listdir(path)
    paths = []
    for name in names:
        mask = os.path.join(path, f"{name[:-4]}_mask.png")
        if os.path.exists(mask):
            paths.append((os.path.join(path, name), mask))
    return paths


class LoadImageDataset(Dataset):
    def __init__(self, filepaths, IMAGE_SIZE, apply_transform=False):

        self.filepaths = filepaths
        self.apply_transform = apply_transform
        self.resize_inp = Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation = InterpolationMode.NEAREST)
        self.resize_mask = Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation = InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.filepaths)

    def transform(self, image, mask):

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        return image, mask

    def __getitem__(self, idx):
        inp_img_path = self.filepaths[idx][0]
        inp_img = self.resize_inp(read_image(inp_img_path, mode=ImageReadMode.RGB))/ 255.0
        out_img_path = self.filepaths[idx][1]
        out_img = self.resize_mask(read_image(out_img_path, mode=ImageReadMode.RGB))[0]/ 255.0

        if self.apply_transform:
            inp_img, out_img = self.transform(inp_img, out_img)
        sample = (inp_img, out_img)
        return sample