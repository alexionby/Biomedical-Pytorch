# BioMedical-PyTorch
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

Additional:
- http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf
- https://arxiv.org/pdf/1606.04797.pdf
- https://jeffknupp.com/blog/2014/06/18/improve-your-python-python-classes-and-object-oriented-programming/
- https://github.com/lanpa/tensorboard-pytorch
- https://arxiv.org/pdf/1505.04597.pdf
- https://tuatini.me/practical-image-segmentation-with-unet/

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

# CHANGEABLE LOSS: 
- f(x) = BCE + 1 - DICE
- f(x) = DICE
- f(x) = BCE

# cyclic learning rate

# DILATED CONV
https://github.com/lyakaap/Kaggle-Carvana-3rd-Place-Solution/blob/master/model.py
https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/40199

# ----
https://towardsdatascience.com/pytorch-tutorial-distilled-95ce8781a89c
https://arxiv.org/abs/1706.06169
