import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import argparse

#my inputs
from model import UNet, UNetConvBlock, UNetUpBlock
from loader import dataloader

parser = argparse.ArgumentParser(description="Set path to trained model and image for prediction")

parser.add_argument('--model', help="Path to model for loading")
parser.add_argument('--image', help="Image for prediction")

args = parser.parse_args()

def main():

    use_cuda = torch.cuda.is_available()
    
    if args.model:
        model = torch.load(args.model)
        if use_cuda:
            model.cuda()
    else:
        return
    
    if args.image:
        image = Image.open(args.image)
    else:
        return

    image = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
        ])(image).unsqueeze_(0)

    #print(image.shape)

    image = Variable(image)
    
    if use_cuda:
        image = image.cuda()

    result = model(image)
    result = F.sigmoid(result)
    
    #print(result[0][0])

    result = transforms.ToPILImage()(result.data[0].cpu())
    result.save('result.png', "PNG")

if __name__ == '__main__':
    main()