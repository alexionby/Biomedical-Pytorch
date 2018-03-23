import time
import random

from torchvision import transforms
from weights import balanced_weights

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

from PIL import Image

aug_dict = {
    'random_crop': transforms.RandomCrop,
    'center_crop': transforms.CenterCrop,
    'vertical_flip': transforms.RandomVerticalFlip,
    'horizontal_flip': transforms.RandomHorizontalFlip,
    'random_rotate': transforms.RandomRotation,
    'color_jitter' : transforms.ColorJitter,
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
            transforms_list.append(aug_dict[augmentation](*aug_values[augmentation]))
        else:
            transforms_list.append(aug_dict[augmentation]())
    
    t = time.time()
    try:
        seed = int(str(t-int(t))[2:])
    except ValueError:
        seed = int(str(t-int(t))[2:-4])
    
    #print(seed, 'here')
    #elastic transforms
    #transforms_list.append(transforms.Lambda(lambda x: elastic_to_torch(x, seed)))

    transforms_list.append(transforms.ToTensor())

    random.seed(seed)
    sample['image'] = transforms.Compose(transforms_list)(sample['image'])

    random.seed(seed)
    sample['mask'] = transforms.Compose(transforms_list)(sample['mask']).byte()

    if weight_function:
        sample['weights'] = weight_function(sample['mask'].float())

    return sample


def elastic_to_torch(image, seed):

    seed = seed % (2**32 -1)
    #print(seed, 'there')
    image = np.asarray(image)
    
    mask = False
    if len(np.unique(image)) == 2:
        mask = True

    image = elastic_transform(image, 112, 12, seed)
    
    if mask:
        image[ image > 0] = 255

    return Image.fromarray(image)

def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)
    else:
        random_state = np.random.RandomState(random_state)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))
    
    return map_coordinates(image, indices, order=1).reshape(shape)


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

