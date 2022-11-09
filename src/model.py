from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    Conv2DTranspose,
    BatchNormalization,
    Dropout,
    Lambda,
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
    p = Dropout(rate=0.3)(p)
    return x, p


def decoder_block(x, skip_features, n_filters):
    x = Conv2DTranspose(n_filters, kernel_size=(2, 2), strides=2, padding="same")(x)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, n_filters)
    return x


def unet_model(input_shape, n_channels_out, n_channels_bottleneck=1024):

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

    outputs = Conv2D(filters=n_channels_out, kernel_size=1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs, name="U-Net")

    return model
