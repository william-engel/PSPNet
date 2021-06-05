import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Conv2D, MaxPool2D, AveragePooling2D, BatchNormalization, Add,
                                     Activation, Input, ZeroPadding2D, Flatten, GlobalAveragePooling2D, Dropout,
                                     UpSampling2D, Concatenate, Reshape, Softmax, ReLU)
from tensorflow.keras.layers.experimental.preprocessing import Resizing

def ConvBlock(x, filters, kernel=(1,1), strides=(1,1), dr=(1,1),
              stage=None, block=None, ident=True):

    '''
        dr: tuple, dilitation_rate, e.g. (1,1)
        ident: bool, if ident == true => downsampling
    '''
    # name
    name = 'conv' + str(stage) + '_block' + str(block)

    # filters
    f1, f2, f3 = filters

    # save inital input value
    x_initial = x

    x = Conv2D(filters=f1, kernel_size=(1,1), strides=strides, name=name+'_1_conv')(x)
    x = BatchNormalization(name=name + '_1_bn')(x)
    x = ReLU(name=name + '_1_relu')(x)

    x = Conv2D(filters=f2, kernel_size=kernel, strides=(1,1), padding='same', dilation_rate=dr, name=name+'_2_conv')(x)
    x = BatchNormalization(name=name + '_2_bn')(x)
    x = ReLU(name=name + '_2_relu')(x)

    x = Conv2D(filters=f3, kernel_size=(1,1), strides=(1,1), name=name+'_3_conv')(x)
    x = BatchNormalization(name=name + '_3_bn')(x)

    # reshape input dim to output dim
    if ident == False:
        x_initial = Conv2D(filters=f3, kernel_size=(1,1), strides=strides, name=name+'_0_conv')(x_initial)
        x_initial = BatchNormalization(name=name + '_0_bn')(x_initial)

    # add the input value x to the output and apply ReLU function
    x = Add(name=name + '_add')([x_initial, x])
    x = ReLU(name=name + '_out')(x)

    return x

def ResNet50(input_shape=None, input_tensor=None, num_classes=None, dr=(1,1)):

    '''
        dr: tuple, dilitation_rate, e.g. (1,1)
    '''

    if input_tensor is None:
        input = Input(shape = input_shape, name = 'input')
    elif tf.keras.backend.is_keras_tensor(input_tensor):
        input = Input(shape = input_tensor.shape[1:], name = 'input')

    # INPUT
    x = ZeroPadding2D(padding = (3,3), name = 'conv1_pad')(input)

    # STAGE 1
    x = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding='valid', name='conv1_conv')(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(name='conv1_relu')(x)
    x = ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = MaxPool2D(pool_size=(3,3), strides=(2, 2), name='pool1_pool')(x)

    # STAGE 2
    x = ConvBlock(x, filters=[64, 64, 256], strides=(1,1), dr=dr, stage=2, block=1, ident=False)
    x = ConvBlock(x, filters=[64, 64, 256], dr=dr, stage=2, block=2)
    x = ConvBlock(x, filters=[64, 64, 256], dr=dr, stage=2, block=3)

    # STAGE 3
    x = ConvBlock(x, filters=[128,128,512], strides=(2,2), dr=dr, stage=3, block=1, ident=False)
    x = ConvBlock(x, filters=[128,128,512], dr=dr, stage=3, block=2)
    x = ConvBlock(x, filters=[128,128,512], dr=dr, stage=3, block=3)
    x = ConvBlock(x, filters=[128,128,512], dr=dr, stage=3, block=4)

    # STAGE 4
    x = ConvBlock(x, filters=[256, 256, 1024], strides=(2,2), dr = dr, stage=4, block=1, ident=False)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=2)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=3)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=4)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=5)
    x = ConvBlock(x, filters=[256, 256, 1024], dr=dr, stage=4, block=6)

    # STAGE 5
    x = ConvBlock(x, filters=[512, 512, 2048], strides=(2,2), dr=dr, stage=5, block=1, ident = False)
    x = ConvBlock(x, filters=[512, 512, 2048], dr=dr, stage=5, block=2)
    x = ConvBlock(x, filters=[512, 512, 2048], dr=dr, stage=5, block=3)

    # OUTPUT
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    output = Dense(units = num_classes,  activation='softmax', name='predictions')(x)

    model = Model(inputs = [input], outputs = [output], name = 'ResNet50')

    return model