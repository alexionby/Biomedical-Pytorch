import os, glob, shutil
import random
from tqdm import tqdm

from sklearn.utils import shuffle
import numpy as np

class DataDescription:    

    common_extensions = ['jpg','tif','png']
    train_path = os.path.join('data','train')
    valid_path = os.path.join('data','valid')

    images_folder = 'images'
    masks_folder = 'masks'
    weights_folder = 'weights'

    train_images_path = os.path.join(train_path, images_folder)
    train_masks_path = os.path.join(train_path, masks_folder)
    train_weights_path = os.path.join(train_path, weights_folder)

    valid_images_path = os.path.join(valid_path, images_folder)
    valid_masks_path = os.path.join(valid_path, masks_folder)
    valid_weights_path = os.path.join(valid_path, weights_folder)
    
    def __init__(self, **kwargs):

        self.img_path = kwargs.get('img_path', os.path.join('data','images'))
        self.mask_path = kwargs.get('mask_path', os.path.join('data','masks'))
        self.weight_path = kwargs.get('weight_path', os.path.join('data','weights'))

        self.images, self.masks, self.weights = self.find_images(self.img_path,
                                                kwargs['img_extensions'] or self.common_extensions, 
                                                self.mask_path,
                                                kwargs['mask_extensions'] or self.common_extensions,
                                                self.weight_path,
                                                kwargs['weight_extensions'] or self.common_extensions)

        if kwargs['valid_split']:
            self.make_split(kwargs['valid_split'], kwargs['valid_shuffle'])
        else:
            self.train_images = self.images
            self.train_masks = self.masks
            self.train_weights = self.weights

            self.valid_images = None
            self.valid_masks = None
            self.valid_weights = None
        
        self.create_dataset()

    @staticmethod
    def create_dir_and_copy(filenames, path_from, path_to):
        try:
            shutil.rmtree(path_to)
            print('Recreating dir')
        except:
            print('Creating new dir')
        os.mkdir(path_to)
        for filename in tqdm(filenames):
            shutil.copyfile(os.path.join(path_from, filename), os.path.join(path_to, filename))
        
    def create_dataset(self):
        
        if self.train_images and self.train_masks:
            DataDescription.create_dir_and_copy(self.train_images, self.img_path, self.train_images_path)
            DataDescription.create_dir_and_copy(self.train_masks, self.mask_path, self.train_masks_path)
            if None not in self.train_weights:
                DataDescription.create_dir_and_copy(self.train_weights, self.weight_path, self.train_weights_path)

        if self.valid_images and self.valid_masks:
            DataDescription.create_dir_and_copy(self.valid_images, self.img_path, self.valid_images_path)
            DataDescription.create_dir_and_copy(self.valid_masks, self.mask_path, self.valid_masks_path)
            if None not in self.valid_weights:
                DataDescription.create_dir_and_copy(self.valid_weights, self.weight_path, self.valid_weights_path)

    def make_split(self, valid_split, valid_shuffle=True):

        if valid_split > 0 and valid_split < 1:
            self.valid_split = valid_split
        else:
            raise ValueError('Split value must be in (0,1) range!')

        if valid_shuffle:
            images, masks, weights = shuffle(self.images, self.masks, self.weights, random_state=0)
            print(images[:15])
            print(masks[:15])
        else:
            images, masks, weights = self.images, self.masks, self.weights
            #images = sorted(images, key=lambda x: int(x.split('.')[0]))
            #masks = sorted(masks, key=lambda x: int(x.split('.')[0]))
            print(images[:15])
            print(masks[:15])

        split_value = int(len(images) * (1 - valid_split))

        self.train_images = images[:split_value]
        self.train_masks = masks[:split_value]
        self.train_weights = weights[:split_value]

        self.valid_images = images[split_value:]
        self.valid_masks = masks[split_value:]
        self.valid_weights = weights[split_value:]

        print('Train size: ', len(self.train_images), len(self.train_masks), len(self.train_weights)) 
        print('Validation size: ', len(self.valid_images), len(self.valid_masks), len(self.valid_weights))

    
    def find_images(self, img_path, img_ext, mask_path, mask_ext, weight_path, weight_ext):
        
        images, masks, weights = [], [], []

        for image_name in tqdm(os.listdir(img_path)):
            if image_name.split('.')[-1].lower() in img_ext:
                sub_name = image_name.split('.')[-2] + '.'
                for mask_name in os.listdir(mask_path):
                    if mask_name.startswith(sub_name) and mask_name.split('.')[-1] in mask_ext:
                        images.append(image_name)
                        masks.append(mask_name)
                        break
                else:
                    raise 'Mask not found'
            
                for weight_name in os.listdir(weight_path):
                    if weight_name.startswith(sub_name) and weight_name.split('.')[-1] in weight_ext:
                        weights.append(weight_name)
                        break
        
        assert len(images) == len(masks), 'Number of images not equal number of masks.'
        
        if len(images) == len(masks) == len(weights):
            return images, masks, weights
        else:
            return images, masks, [None] * len(images)

def main():
    """Tests should be here"""
    DataDescription()

if __name__ == '__main__':
    main()