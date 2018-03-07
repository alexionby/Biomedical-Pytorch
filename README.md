# Pytorch-Unet-BSMU

Inspiration:
- https://github.com/meetshah1995/pytorch-semseg
- https://github.com/jaxony/unet-pytorch
- https://github.com/milesial/Pytorch-UNet
- https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py

Ultrasound inspiration:
- https://github.com/jocicmarko/ultrasound-nerve-seg
- https://www.slideshare.net/Eduardyantov/ultrasound-segmentation-kaggle-review
- https://github.com/yihui-he/Ultrasound-Nerve-Segmentation
- https://github.com/EdwardTyantov/ultrasound-nerve-segmentation

TODO:

- Add Elastic Transforms
- Add Color augmentations
- Add CRF

#AUGMENTATIONS
https://github.com/aleju/imgaug

# MODEL MEMORY CAP
https://github.com/jacobkimmel/pytorch_modelsize

# GPU MEMORY USAGE
https://github.com/anderskm/gputil

# ACCUMULATE GRADIENTS
# from https://discuss.pytorch.org/t/pytorch-gradients/884/10
optimizer.zero_grad()

for i in range(minibatch):
    loss = model(batch_data[i])
    loss.backward()

optimizer.step()

# CLEVER WEIGHTS IN BCELOSS

# CLEVER LOSS LIKE: f(x) = BCE + 1 - DICE

# cyclic learning rate

