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

#my import
from description import DataDescription
from weights import balanced_weights

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

aug_dict = {
    'random_crop': transforms.RandomCrop,
    'center_crop': transforms.CenterCrop,
    'vertical_flip': transforms.RandomVerticalFlip,
    'horizontal_flip': transforms.RandomHorizontalFlip,
    'random_rotate': transforms.RandomRotation,
    'resize': transforms.Resize,
}

def transform(sample,
              weight_function=False,
              aug_order = [],
              aug_values = {},
              ):
    
    transforms_list = []

    #args represent the order
    for augmentation in aug_order:
        if augmentation in aug_values:
            transforms_list.append(aug_dict[augmentation](aug_values[augmentation]))
        else:
            transforms_list.append(aug_dict[augmentation])
    
    transforms_list.append(transforms.ToTensor())

    t = time.time()
    try:
        seed = int(str(t-int(t))[2:])
    except ValueError:
        seed = int(str(t-int(t))[2:-4])

    random.seed(seed)
    sample['image'] = transforms.Compose(transforms_list)(sample['image'])

    random.seed(seed)
    sample['mask'] = transforms.Compose(transforms_list)(sample['mask']).byte()

    if weight_function:
        sample['weights'] = weight_function(sample['mask'].float())

    return sample

class UnetDataset(Dataset, DataDescription):
    """Unet images and masks"""

    def __init__(self,
                 img_channels = 3,
                 img_ext = DataDescription.common_extensions,
                 img_path = os.path.join('data','images'),
                 mask_channels = 1, 
                 mask_ext = DataDescription.common_extensions,
                 mask_path = os.path.join('data','masks'),
                 common_length=5, #None,
                 valid_split = 0.25, #None,
                 valid_shuffle = True,
                 transform=None,
                 weight_function=None,
                 aug_order=[],
                 aug_values={}
                 ):
        """
        Args:
            train (boolean): Shows whether it's trainable images or not
            img_grayscale (boolean): "RGB" or "GR img_gray=self.img_grayAY"
            transform (callable, optional): Optional transform to be applied
                on images and masks sample.
        """

        DataDescription.__init__(self,
                                 img_ext,
                                 img_path,
                                 mask_ext,
                                 mask_path,
                                 common_length,
                                 valid_split,
                                 valid_shuffle)
        
        #print('channels', self.img_channels)
        print(self.train_images_path)

        self.img_gray = True if img_channels == 1 else False
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
                            num_workers=4)

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