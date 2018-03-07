from __future__ import print_function, division
import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
#import cv2

from skimage.morphology import binary_erosion, binary_dilation
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray

import time

from matplotlib import pyplot as plt

from torchvision.datasets.folder import IMG_EXTENSIONS
IMG_EXTENSIONS.append('tiff')
IMG_EXTENSIONS.append('tif')

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""
def __getitem__(self,index):      
    img = Image.open(self.data[index]).convert('RGB')
    target = Image.open(self.data_labels[index])
    
    seed = np.random.randint(2147483647) # make a seed with numpy generator 
    random.seed(seed) # apply this seed to img tranfsorms
    if self.transform is not None:
        img = self.transform(img)
        
    random.seed(seed) # apply this seed to target tranfsorms
    if self.target_transform is not None:
        target = self.target_transform(target)

    target = torch.ByteTensor(np.array(target))

    return img, target
"""

def transform(sample, img_gray=False, crop_in=512, crop_out=512):

    seed = np.random.randint(2147483647)

    random.seed(seed)
    sample['image'] = transforms.Compose([
        transforms.RandomCrop((crop_in, crop_in)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # maybe bug here
        #transforms.Lambda( lambda x: transforms.Grayscale(1)(x) if img_gray else x ),
        transforms.ToTensor(),
    ])(sample['image'])

    random.seed(seed)
    sample['mask'] = transforms.Compose([
        transforms.RandomCrop((crop_in, crop_in)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop((crop_out, crop_out)),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])(sample['mask']).byte()

    binary_mask = sample['mask'].numpy().squeeze()
    weights = np.ones( binary_mask.shape, dtype=float) * 0.5

    binary_eroded = binary_mask.copy()
    for pad in range(6):
        mask_boundary = binary_eroded - binary_erosion(binary_eroded)
        binary_eroded = binary_eroded - mask_boundary
        weights[ mask_boundary > 0 ] = 1.0 - 0.1 * pad
    
    binary_dilated = binary_mask.copy()
    for pad in range(6):
        mask_boundary = binary_dilation(binary_dilated) - binary_dilated
        binary_dilated = binary_dilated + mask_boundary
        weights[ mask_boundary > 0 ] = 1.0 - 0.1 * pad
    
    #print(weights.min())
    #img = Image.fromarray(np.uint8(weights * 255))
    #img.save('result.png',"PNG")

    sample['weights'] = torch.FloatTensor(weights).unsqueeze_(0)

    return sample

class UnetDataset(Dataset):
    """Unet images and masks"""

    def __init__(self, train=True, img_gray=False, transform=None):
        """
        Args:
            train (boolean): Shows whether it's trainable images or not
            img_grayscale (boolean): "RGB" or "GRAY"
            transform (callable, optional): Optional transform to be applied
                on images and masks sample.
        """
        self.img_gray = img_gray
        self.folder = "train" if train else "val"
        self.images_path = os.path.join("data", self.folder, "images")
        self.masks_path  = os.path.join("data", self.folder, "masks")

        #print("folders:", self.images_path, self.masks_path)
        #print("sizes:", len(os.listdir(self.images_path)), len(os.listdir(self.masks_path)))
        
        assert len(os.listdir(self.images_path)) ==  len(os.listdir(self.masks_path))

        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.images_path))

    def __getitem__(self, idx):
        img_name = os.path.join(self.images_path, os.listdir(self.images_path)[idx])

        # WEIRD CODE ALERT!!! 
        mask_name = os.path.join(self.masks_path, os.listdir(self.images_path)[idx][:-3] + 'tif')

        image = Image.open(img_name)
        mask  = Image.open(mask_name)
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample, img_gray=self.img_gray)

        return sample

def dataloader(batch_size=2, crop_in=512, crop_out=512):

    transformed_dataset = UnetDataset(True, False, transform=transform)
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=4)
    
    return dataloader


# For Tests

def main():
    transformed_dataset = UnetDataset(True, True, transform=transform)

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['mask'].size())
        out = torch.cat((sample['image'],sample['mask']), 2)
        print(out.shape)
        out = utils.make_grid(out)
        print(out.shape)
        out = out.numpy().transpose((1, 2, 0))
        print(out.shape)
        plt.imshow(out)
        plt.show()

        if i == 3:
            break

    print("Dataloader:")
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)
    
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
            sample_batched['mask'].size())

        if i_batch == 3:
            break

if __name__ == '__main__':
    main()