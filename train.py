#my inputs
from model import UNet, UNetConvBlock, UNetUpBlock
from loader import dataloader, UnetDataset, transform
from losses import SoftDiceLoss
from description import DataDescription
from weights import balanced_weights

# PyTorch
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

# Def. inputs
import os
from PIL import Image
import argparse
from tqdm import tqdm
import time

parser = argparse.ArgumentParser(description='Setting model parameters')

parser.add_argument('--depth', default=5, type=int, help="Unet depth")
parser.add_argument('--n','--n_filters', type=int, default=6, help="2**N filters on first Layer")
parser.add_argument('--ch', type=int, default=1, help="Num of channels")
parser.add_argument('--cl','--n_classes', default=1, type=int, help="number of output channels(classes)")
parser.add_argument('--pad', type=bool, default=True, help="""if True, apply padding such that the input shape
                                                                is the same as the output.
                                                                This may introduce artifacts""")
parser.add_argument('--bn', type=bool, default=True, help="""Use BatchNorm after layers with an
                                                                        activation function""")
parser.add_argument('--up_mode', default='upconv', choices=['upconv', 'upsample'], help="""One of 'upconv' or 'upsample'.
                                                                            'upconv' will use transposed convolutions for
                                                                            learned upsampling.
                                                                            'upsample' will use bilinear upsampling.""")
parser.add_argument('--model', help="Path to model for loading")
parser.add_argument('--batch_size', help="Size of batch", default=4, type=int )
parser.add_argument('--crop_size', help="Size of subimage for random crop", type=int, default=512)

args = parser.parse_args()
print(args)

input_data = {
    'img_extensions' : None,
    'img_path' : 'data/images',
    'mask_extenstions' : None,
    'mask_path' : 'data/masks',
    'common_length' : None,
    'valid_split' : None,
    'valid_shuffle' : False,

    'img_channels' : None,
    'mask_channels' : None,
    
}

def main():
    
    if args.model:
        model = torch.load('model.pt')
    else:
        model = UNet(in_channels=args.ch,
                     n_classes=args.cl,
                     depth=args.depth,
                     wf=args.n,
                     padding=args.pad,
                     batch_norm=args.bn,
                     up_mode=args.up_mode
                    )

    dataset = UnetDataset(img_channels=1, #? for what this is here?
                          transform=transform, 
                          weight_function=balanced_weights,
                          aug_order=['random_crop'],
                          aug_values={'random_crop': 512})

    model.cuda()

    # Observe that all parameters are being optimized
    optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True, factor=0.25)

    epochs = 150
    for epoch in range(epochs):

        epoch_loss = 0
        print("epoch: ", epoch, '/', epochs)

        datagen = dataloader(dataset, batch_size=args.batch_size)

        for i_batch, sample_batched in tqdm(enumerate(datagen)):

            inputs = Variable(sample_batched['image']).cuda()
            labels = Variable(sample_batched['mask']).cuda()
            weights = Variable(sample_batched['weights']).cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            #print(inputs.shape, labels.shape)
            #print(outputs.shape)
            
            loss = F.binary_cross_entropy_with_logits(outputs, labels.float(), weight=weights) 
            #loss = criterion(outputs, labels.float())

            epoch_loss += loss.data[0]

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
        
        print("train loss:", epoch_loss / (i_batch + 1))

        del loss, outputs, labels, inputs, weights
        torch.cuda.empty_cache()

        dataset.switch_mode()
        datagen = dataloader(dataset, batch_size=args.batch_size)

        for i_batch, sample_batched in tqdm(enumerate(datagen)):

            inputs = Variable(sample_batched['image']).cuda()
            labels = Variable(sample_batched['mask']).cuda()
            weights = Variable(sample_batched['weights']).cuda()

            optimizer.zero_grad()
            outputs = model(inputs)

            probs = F.sigmoid(outputs)

            loss = F.binary_cross_entropy_with_logits(outputs, labels.float(), weight=weights) 
            epoch_loss += loss.data[0]

            #loss.backward()
            #optimizer.step()

            if i_batch == 2:

                im = torchvision.transforms.ToPILImage()(probs.data[0].cpu())
                im.save("learn/pred/" + str(epoch) + "_final.jpg" , "JPEG")

                im = torchvision.transforms.ToPILImage()(sample_batched['mask'][0].float())
                im.save("learn/mask/" + str(epoch) + "_final.jpg" , "JPEG")
                
                im = torchvision.transforms.ToPILImage()(sample_batched['image'][0])
                im.save("learn/image/" + str(epoch) + "_final.jpg", "JPEG")
    
            del loss, probs, outputs, inputs, labels
            torch.cuda.empty_cache()

        scheduler.step(epoch_loss / (i_batch + 1))
        print("valid loss:", epoch_loss / (i_batch + 1))
        torch.save(model, 'model.pt')

        dataset.switch_mode()

if __name__ == '__main__':
    main()