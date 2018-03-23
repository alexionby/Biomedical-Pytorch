from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import utils

from PIL import Image
#import cv2

#from skimage.morphology import binary_erosion, binary_dilation
#from skimage.exposure import rescale_intensity
#from skimage.color import rgb2gray

from torchvision.datasets.folder import IMG_EXTENSIONS
IMG_EXTENSIONS.append('tiff')
IMG_EXTENSIONS.append('tif')

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#my import
from description import DataDescription
from transforms import transform

class UnetDataset(Dataset, DataDescription):
    """Unet images and masks"""

    def __init__(self,
                 transform=None,
                 weight_function=None,
                 aug_order=[],
                 aug_values={},
                 **kwargs
                 ):

        print(kwargs)

        DataDescription.__init__(self, **kwargs)

        print(self.train_images_path)

        self.transform = transform
        self.aug_order = aug_order
        self.aug_values = aug_values
        self.weight_function = weight_function
        self.is_train = True

    def __len__(self):
        if self.is_train:
            return len(os.listdir(self.train_images_path))
        else:
            return len(os.listdir(self.valid_images_path))

    def __getitem__(self, idx):

        if self.is_train:
            img_name = os.path.join(self.train_images_path, self.train_images[idx])
            mask_name = os.path.join(self.train_masks_path, self.train_masks[idx])
        else:
            img_name = os.path.join(self.valid_images_path, self.valid_images[idx])
            mask_name = os.path.join(self.valid_masks_path, self.valid_masks[idx])

        image = Image.open(img_name)
        mask  = Image.open(mask_name)

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample, 
                                    weight_function=self.weight_function, 
                                    aug_order=self.aug_order,
                                    aug_values=self.aug_values)

        return sample

    
    def switch_mode(self):
        self.is_train = not self.is_train

def dataloader(dataset, batch_size=2, crop_in=512, crop_out=512):

    #transformed_dataset = UnetDataset(transform=transform)
    datagen = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=8)

    return datagen


# For Tests

def main():

    transformed_dataset = UnetDataset(transform=transform)

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['mask'].size())
        out = torch.cat((sample['image'],sample['mask'].float()), 2)
        print(out.shape)
        out = utils.make_grid(out)
        print(out.shape)
        out = out.numpy().transpose((1, 2, 0))
        print(out.shape)
        plt.imshow(out)
        plt.show()

        if i == 3:
            break
    
    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['mask'].size())
        out = torch.cat((sample['image'],sample['mask'].float()), 2)
        print(out.shape)
        out = utils.make_grid(out)
        print(out.shape)
        out = out.numpy().transpose((1, 2, 0))
        print(out.shape)
        plt.imshow(out)
        plt.show()

        if i == 2:
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