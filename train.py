#my inputs
from model import UNet, UNetConvBlock, UNetUpBlock
from loader import dataloader

# PyTorch
import torch
from torch import nn
from torch.optim import SGD
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

# Def. inputs
import os
from PIL import Image


def main():
    
    if 'model.pt' in os.listdir('.'):
        model = torch.load('model.pt')
    else:
        model = UNet(batch_norm=True)

    model.cuda()

    criterion = nn.BCEWithLogitsLoss()

    # Observe that all parameters are being optimized
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(300):

        epoch_loss = 0
        print("epoch: ", epoch)

        for i_batch, sample_batched in enumerate(dataloader(batch_size=3)):

            inputs = Variable(sample_batched['image']).cuda()
            labels = Variable(sample_batched['mask']).cuda()

            optimizer.zero_grad()

            outputs = model(inputs)

            probs = F.sigmoid(outputs)

            loss = criterion(outputs, labels)

            epoch_loss += loss.data[0]

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            #if i_batch == 2:
            #    break
        


        im = torchvision.transforms.ToPILImage()(probs.data[0].cpu())
        im.save("learn/pred/" + str(epoch) + "_final.jpg" , "JPEG")

        im = torchvision.transforms.ToPILImage()(sample_batched['mask'][0])
        im.save("learn/mask/" + str(epoch) + "_final.jpg" , "JPEG")
        
        im = torchvision.transforms.ToPILImage()(sample_batched['image'][0])
        im.save("learn/image/" + str(epoch) + "_final.jpg", "JPEG")
        
        print("epoch loss:", epoch_loss / (i_batch + 1))
        torch.save(model, 'model.pt')

if __name__ == '__main__':
    main()