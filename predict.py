#my inputs
from model import UNet, UNetConvBlock, UNetUpBlock
from loader import dataloader

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Set path to trained model and image for prediction')

parser.add_argument('--model', help="Path to model for loading")
parser.add_argument('--image', help="Image for prediction")

args = parser.parse_args()

def main():
    
    if args.model:
        model = torch.load(args.model)
        model.cuda()
    else:
        return
    
    if args.image:
        image = Image.open(args.image)
    else:
        return 

    image = transforms.Compose([
        transforms.RandomCrop((800,800)),
        transforms.ToTensor()
        ])(image).unsqueeze_(0)

    print(image.shape)

    image = Variable(image).cuda()

    result = model(image)
    result = F.sigmoid(result)
    print(result[0][0])

    result = transforms.ToPILImage()(result.data[0].cpu())
    result.save('result.png', "PNG")

if __name__ == '__main__':
    main()