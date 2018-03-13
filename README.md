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
- Add Dropout ( and model.eval() )

#DOUBLE CONV LAYER?

    def double_conv_layer(x, size, dropout, batch_norm):
        if K.image_dim_ordering() == 'th':
            axis = 1
        else:
            axis = 3
        conv = Conv2D(size, (3, 3), padding='same')(x)
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(size, (3, 3), padding='same')(conv)
        if batch_norm is True:
            conv = BatchNormalization(axis=axis)(conv)
        conv = Activation('relu')(conv)
        if dropout > 0:
            conv = SpatialDropout2D(dropout)(conv)
    return conv

    def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

    def jacard_coef(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


    def jacard_coef_loss(y_true, y_pred):
        return -jacard_coef(y_true, y_pred)


    def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

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
