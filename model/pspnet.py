import sys

sys.path.insert(1, "C:\\Users\\willi\\Desktop\\William\\03_Weiterbildung\\08_KI\\02_PSPNet\\PSPNet")

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPool2D, AveragePooling2D, BatchNormalization, Add,
                                     Activation, Input, ZeroPadding2D, Flatten, GlobalAveragePooling2D, Dropout,
                                     UpSampling2D, Concatenate, Reshape, Softmax, ReLU)
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.optimizers import Adam


from model.resnet50 import ResNet50

def PyramidPooling(input_shape = None, input_tensor = None, pool_sizes = None):
    '''pool_sizes = [(1,1), (2,2), (3,3), (6,6)]'''

    if input_tensor is None:
        input = Input(shape = input_shape)
    elif tf.keras.backend.is_keras_tensor(input_tensor):
        input = Input(shape = input_tensor.shape[1:])

    _, input_height, input_width, input_depth = input.shape

    N = len(pool_sizes) # number pyramid levels

    f = input_depth // N # pyramid level depth
    
    output = [input]

    for level, pool_size in enumerate(pool_sizes):
        x = AveragePooling2D(pool_size = pool_size, padding = 'same', name='avg_'+str(level))(input)
        x = Conv2D(filters = f, kernel_size = (1,1), name='conv_'+str(level))(x)
        x = UpSampling2D(size = pool_size, interpolation = 'bilinear', name='bilinear_'+str(level))(x)

        # Crop/pad to input size
        # Pooling with padding = 'same': After upsamling => size > input size => crop
        # Pooling with padding = 'valid': After upsamling => size < input size => pad
        height, width = x.shape[1:3]
        dif_height, dif_width = height - input_height, width - input_width
        x = tf.keras.layers.Cropping2D(cropping = ((0,dif_height), (0,dif_width)), name='crop_'+str(level))(x)

        output += [x]

    # concatenate input and pyramid levels
    output = Concatenate(name='concatenate')(output)

    model = Model(inputs = [input], outputs = [output], name = 'pyramid_module')

    return model

def Decoder(input_shape = None, input_tensor = None, target_shape = None, num_classes = None):
    '''target_shape: (H,W)'''

    if input_tensor is None:
        input = Input(shape = input_shape)
    elif tf.keras.backend.is_keras_tensor(input_tensor):
        input = Input(shape = input_tensor.shape[1:])

    height, width = target_shape

    x = Conv2D(filters = 512, kernel_size = (3,3))(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Dropout(.1)(x)

    x = Conv2D(filters = num_classes, kernel_size = (1,1))(x)
    x = Resizing(height = height, width = width, interpolation = 'bilinear', name = 'resize')(x)
    
    output = Softmax(name = 'segmentation')(x)

    model = Model(inputs = [input], outputs = [output], name = 'decoder')

    return model

def PSPNet(input_shape, num_classes, pool_sizes):

    input = Input(shape = input_shape, name = 'input')

    target_shape = input_shape[:2] # (H,W,B) => (H,W)

    # truncate resnet after 3. stage
    resnet = ResNet50(input_tensor=input, num_classes=num_classes, dr=(2,2))
    resnet_output = resnet.get_layer('conv3_block4_out')
    basemodel = Model(inputs = resnet.input, outputs = resnet_output.output, name = 'resnet')

    pyramid_module = PyramidPooling(input_tensor=basemodel.output, pool_sizes=pool_sizes)

    decoder = Decoder(input_tensor = pyramid_module.output, target_shape = target_shape, num_classes = num_classes)

    x = basemodel(input)
    x = pyramid_module(x)
    output = decoder(x)

    pspnet = Model(inputs = [input], outputs = [output], name = 'pspnet')

    return pspnet