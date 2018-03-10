import os, glob
import random
from sklearn.utils import shuffle

import numpy as np

"""
    X = np.array([[1., 0.], [2., 1.], [0., 0.]])
    y = np.array([0, 1, 2])
    X, y = shuffle(X, y, random_state=0)
"""


class DataDescription(object):    

    common_extensions = ['jpg','tif','png']
    
    def __init__(self, 
                 img_channels = 3,
                 img_ext = common_extensions,
                 img_path = 'data/train/images',
                 mask_channels = 1, 
                 mask_ext = common_extensions,
                 mask_path = 'data/train/masks',
                 common_length=-14, #None,
                 valid_split = 0.25 #None,
                 ):

        self.img_channels = img_channels
        #self.img_ext = img_ext
        #self.img_path = img_path
        self.mask_channels = mask_channels
        #self.mask_ext = mask_ext
        #self.mask_path = mask_path
        #self.common_length = common_length
        
        self.images, self.masks = self.find_images(img_path, 
                                                   img_ext, 
                                                   mask_path, 
                                                   mask_ext, 
                                                   common_length)

        if valid_split:
            if valid_split > 0 and valid_split < 1:
                self.make_split(valid_split)

        

    def make_split(self, valid_split=0):
        X, y = shuffle(self.images, self.masks, random_state=0)
        print(X[:10], y[:10])
    
    def find_images(self, img_path, img_ext, mask_path, mask_ext, common_length):
        
        images = []
        masks = []

        for image_name in os.listdir(img_path)[:100]:
            if image_name.split('.')[-1].lower() in img_ext:
                for mask_name in glob.glob( os.path.join(mask_path , image_name[:common_length].split('.')[0] + '*')):
                    if mask_name.split('.')[-1].lower() in mask_ext:
                        images.append(image_name)
                        masks.append(mask_name.split('/')[-1])
                        break

        return images, masks

def main():
    test = DataDescription()
    

if __name__ == '__main__':
    main()