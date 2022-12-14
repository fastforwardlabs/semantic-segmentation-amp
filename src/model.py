# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2022
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    Activation,
)


def conv_block(x, n_filters):
    x = Conv2D(n_filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(n_filters, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x


def encoder_block(x, n_filters):
    x = conv_block(x, n_filters)
    p = MaxPooling2D(pool_size=(2, 2))(x)
    p = Dropout(rate=0.25)(p)
    return x, p


def decoder_block(x, skip_features, n_filters):
    x = Conv2DTranspose(n_filters, kernel_size=(2, 2), strides=2, padding="same")(x)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, n_filters)
    return x


def unet_model(input_shape, n_channels_out, n_channels_bottleneck=1024):
    """
    Implementation of Unet architecture in Keras

    https://arxiv.org/pdf/1505.04597.pdf

    Args:
        input_shape (tuple)
        n_channels_out (int) - number of channels in ouputer layer
        n_channels_bottleneck (int, optional) - number of channels in bottleneck layer


    """

    input_shape = input_shape + (3,)
    inputs = Input(input_shape)

    # encoder downsampling (x4)
    c1, p1 = encoder_block(inputs, n_channels_bottleneck / 16)
    c2, p2 = encoder_block(p1, n_channels_bottleneck / 8)
    c3, p3 = encoder_block(p2, n_channels_bottleneck / 4)
    c4, p4 = encoder_block(p3, n_channels_bottleneck / 2)

    # bottleneck
    b = conv_block(p4, n_channels_bottleneck)

    # decoder upsampling (x4)
    d1 = decoder_block(b, c4, n_channels_bottleneck / 2)
    d2 = decoder_block(d1, c3, n_channels_bottleneck / 4)
    d3 = decoder_block(d2, c2, n_channels_bottleneck / 8)
    d4 = decoder_block(d3, c1, n_channels_bottleneck / 16)

    outputs = Conv2D(
        filters=n_channels_out, kernel_size=1, padding="same", activation="softmax"
    )(d4)

    model = Model(inputs, outputs, name="U-Net")

    return model
