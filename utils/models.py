# Revised by Sayim Gokyar Jan, Jun, July 2022, Feb 2023
from __future__ import print_function, division

import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, Conv3DTranspose, concatenate

def unet_3d_model(input_size, stages, featurelength, ReconActivation = 'relu'):
    nfeatures = [2**ff*featurelength for ff in np.arange(stages)]
    depth = len(nfeatures)
    conv_ptr = []
    inputs = Input(input_size)
    down = inputs
     
    for depth_cnt in range(depth):
        conv = Conv3D(filters=nfeatures[depth_cnt], kernel_size = 3, padding='same', activation='relu', kernel_initializer='he_normal')(down)
        conv = Conv3D(filters=nfeatures[depth_cnt], kernel_size = 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
        conv_ptr.append(conv)

        # maxpooling
        if depth_cnt < depth-1: # If size of input is odd, only do a 3x3 max pool, in our case it should be good for all stages (128/64/32/16)
            xres = conv.shape.as_list()[1]
            if (xres % 2 == 0):
                pooling_size = (2,2,2)
            elif (xres % 2 == 1):
                pooling_size = (3,3,3)

            down = MaxPooling3D(pool_size=pooling_size)(conv)

    # step up convolutional layers
    for depth_cnt in range(depth-2,-1,-1):

        deconv_shape = conv_ptr[depth_cnt].shape.as_list()
        deconv_shape[0] = None

        # If size of input is odd, then do a 3x3 deconv
        if (deconv_shape[2] % 2 == 0):
            unpooling_size = (2,2,2)
        elif (deconv_shape[2] % 2 == 1):
            unpooling_size = (3,3,3)

        up = concatenate([Conv3DTranspose(nfeatures[depth_cnt], kernel_size = 3, padding='same', strides=unpooling_size)(conv), conv_ptr[depth_cnt]], axis=-1)
        conv = Conv3D(nfeatures[depth_cnt], kernel_size = 3, padding='same', activation='relu', kernel_initializer='he_normal')(up)
        conv = Conv3D(nfeatures[depth_cnt], kernel_size = 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv)
    # Final Stage - combine features by using (1,1,1,1) convolution
    recon = Conv3D(1, (1,1,1), padding='same', activation=ReconActivation)(conv)
    model = Model(inputs=[inputs], outputs=[recon])
    return model

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

