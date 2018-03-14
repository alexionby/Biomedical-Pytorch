import torch
import numpy as np

from skimage.morphology import binary_erosion, binary_dilation
from skimage.morphology import label, disk

def simple_weights(tensor):
    return torch.ones(tensor.shape)

def balanced_weights(tensor):

    #torch.unique() in 0.4.0
    unique_labels = np.unique(tensor.numpy())
    total = tensor.view(-1).shape[0]
    weights = torch.ones(tensor.shape)

    for label in unique_labels:
        label = torch.FloatTensor([int(label)])
        # 1 - freq?
        weights[ tensor == label ] = 1 - (tensor == label).nonzero().shape[0] / total
    
    return weights.unsqueeze_(0)

def strong_binary_borders(tensor, reduce_to = 0.5, reduce_step=0.1):

    binary_mask = tensor.squeeze().numpy()
    weights = np.ones(binary_mask.shape) * reduce_to

    binary_eroded = binary_mask.copy()
    for pad in range(int(reduce_to / reduce_step) + 1):
        mask_boundary = binary_eroded - binary_erosion(binary_eroded)
        binary_eroded = binary_eroded - mask_boundary
        weights[ mask_boundary > 0 ] = 1.0 - reduce_step * pad
    
    binary_dilated = binary_mask.copy()
    for pad in range(int(reduce_to / reduce_step) + 1):
        mask_boundary = binary_dilation(binary_dilated) - binary_dilated
        binary_dilated = binary_dilated + mask_boundary
        weights[ mask_boundary > 0 ] = 1.0 - reduce_step * pad
    
    return torch.FloatTensor(weights).unsqueeze_(0)

def unet_paper_weights(tensor, w0=10, sigma=5, radius=9):

    # 10 * np.exp(-((1+1)**2)/(2*5**2))
    mask = tensor.numpy()
    mask = np.pad(mask, radius, mode='constant', constant_values=(0))
    mask = label(mask)
    for row in mask[radius:-radius, :]:
        for elem in row[radius:-radius]:
            if elem == 0:
                d1 = 0
                d2 = 0
                for r in range(1,radius):
                    print(disk(r))
                    print(disk(r).shape)
                
                return

    return torch.FloatTensor(mask)

def main():
    x = (torch.rand((5,5))).round()
    print(x)
    print(balanced_weights(x))
    print(simple_weights(x))
    print(strong_binary_borders(x))
    print(unet_paper_weights(x))

if __name__ == '__main__':
    main()